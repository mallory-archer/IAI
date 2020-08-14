# examine effect on player by stack size (create loop)

# ------ Fit prospect theory model -----
import random
import numpy as np
import pandas as pd
from statsmodels.base.model import GenericLikelihoodModel
from prospect_theory_funcs import *
from assumption_calc_functions import *
from params import exp_loss_seat_dict, exp_win_seat_dict
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import copy


class Option:
    def __init__(self, name, outcomes):
        self.name = name
        self.outcomes = outcomes

    def calc_option_value(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f):
        return sum([calc_outcome_value(
            calc_outcome_v(x_f=t_outcome['payoff'], alpha_f=alpha_f, lambda_f=lambda_f, beta_f=beta_f),
            calc_outcome_pi(p_f=t_outcome['prob'], c_f=gamma_f if (t_outcome['payoff'] >= 0) else delta_f))
                    for _, t_outcome in self.outcomes.items()])


class ChoiceSituation:
    def __init__(self, sit_options, sit_choice=None, slansky_strength=None, stack_rank=None, post_loss=None):
        self.options = sit_options
        self.option_names = [x.name for x in sit_options]
        self.choice = sit_choice
        self.slansky_strength = slansky_strength
        self.stack_rank = stack_rank
        self.post_loss = post_loss
        self.values = None
        self.prob_options = None
        self.prob_choice = None
        self.pred_choice = None
        if (self.choice is not None) and self.choice not in self.option_names:
            print("Warning: specified choice of '%s' is not an option in choice set: %s" % (
            self.choice, self.option_names))

    def get_option_values(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f):
        return [x.calc_option_value(alpha_f, lambda_f, beta_f, gamma_f, delta_f) for x in self.options]

    def get_option_probs(self, phi_f):
        if self.values is not None:
            return calc_prob_options(self.values, phi_f)
        else:
            print("Error: no option values stored for choice situation. Set via method 'get_option_values(args)' first")
            return None

    def get_choice_location(self, choice_f):
        return self.option_names.index(choice_f)

    def get_prob_choice(self, choice_f):
        if self.prob_options is not None:
            return self.prob_options[self.get_choice_location(choice_f)]
        else:
            print(
                "Error: no option probabilities stored for choice situation. Set via method 'get_option_probs(args)' first")
            return None

    def set_model_values(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f, phi_f):
        self.values = self.get_option_values(alpha_f, lambda_f, beta_f, gamma_f, delta_f)
        self.prob_options = self.get_option_probs(phi_f)
        self.prob_choice = self.get_prob_choice(self.choice)

    def predict_choice(self):
        self.pred_choice = self.option_names[self.prob_options.index(max(self.prob_options))]


class ProspectModel(GenericLikelihoodModel):
    def __init__(self, endog, **kwds):
        super(ProspectModel, self).__init__(endog, **kwds)
        self.alpha_f = None
        self.lambda_f = None
        self.beta_f = None
        self.gamma_f = None
        self.delta_f = None
        self.phi_f = None

    # refresh choice sitaution values under updated model parameters
    def refresh_choice_situation_values(self):
        for t_choice_sit in self.choice_situations_f:
            t_choice_sit.set_model_values(alpha_f=self.alpha_f, lambda_f=self.lambda_f, beta_f=self.beta_f,
                                          gamma_f=self.gamma_f, delta_f=self.delta_f,
                                          phi_f=self.phi_f)

    # ---- Manual LL and score ----
    # replaces default loglikelihood for each observation
    def loglikeobs(self, params):
        # print(self.choice_probs)
        self.alpha_f = params[0]
        self.lambda_f = params[1]
        if self.constrain_beta:
            self.beta_f = self.alpha_f  # self.alpha_f #### set equal to self.alpha_f rather than params[2]
            self.gamma_f = params[2]    #### decrease index by 1
            self.delta_f = params[3]    #### decrease index by 1
            self.phi_f = params[4]      #### decrease index by 1
        else:
            self.beta_f = params[2]
            self.gamma_f = params[3]
            self.delta_f = params[4]
            self.phi_f = params[5]

        # -- for given evaluation of likelihood at current value of params, calculate choice information
        self.refresh_choice_situation_values()

        # --- calc log-likelihood
        LL_f = [np.log(x.prob_choice) for x in self.choice_situations_f]

        return np.array(LL_f)

    # replaces default gradient loglikelihood for each observation (Jacobian)
    def score_obs(self, params, **kwds):
        def calc_grad(options_ff, j_chosen_ff, prob_choice_ff, alpha_ff, lambda_ff, beta_ff, gamma_ff, delta_ff, phi_ff):
            # d(LL)/d(x) = sum[(1/Pr) * d(Pr)/d(x)]
            # d(Pr)/d(x=alpha, beta, lambda) = d(Pr)/d(V) * d(V)/d(v) * d(v)/d(x=alpha, beta, lambda)
            # d(Pr)/d(x=c=gamma, delta)      = d(Pr)/d(V) * d(V)/d(pi) * d(pi)/d(x=c=gamma, delta)
            def d_v_d_alpha(x_fff, alpha_fff):
                return (x_fff ** alpha_fff) * np.log(x_fff) if (x_fff > 0) else 0

            def d_v_d_beta(x_fff, beta_fff, lambda_fff):
                return (-lambda_fff) * ((-x_fff) ** beta_fff) * np.log(-x_fff) if (x_fff < 0) else 0

            def d_v_d_lambda(x_fff, beta_fff):
                return -1 * ((-x_fff) ** beta_fff) if (x_fff < 0) else 0

            def d_pi_d_c(p_fff, c_fff):
                pc = p_fff ** c_fff
                pc1 = (1 - p_fff) ** c_fff
                t_mult = pc * ((pc + pc1) ** (-1 / c_fff))
                t_n1 = np.log(pc + pc1)
                t_d1 = c_fff ** 2
                t_n2 = (pc * np.log(p_fff)) + (pc1 * np.log(1 - p_fff))
                t_d2 = c_fff * (pc + pc1)
                t_quant = (t_n1 / t_d1) - (t_n2 / t_d2)
                return t_mult * (t_quant + np.log(p_fff))

            def d_Vcomponent_d_x(prob_ffff, payoff_ffff, select_param_ffff, alpha_ffff, lambda_ffff, beta_ffff,
                                 gamma_ffff, delta_ffff):
                # for each parameter, returns d_v_d_parameter, d_pi_d_parameter
                if select_param_ffff == 'alpha':
                    if self.constrain_beta:
                        return d_v_d_alpha(x_fff=payoff_ffff, alpha_fff=alpha_ffff) + d_v_d_beta(x_fff=payoff_ffff, beta_fff=beta_ffff, lambda_fff=lambda_ffff), 0
                    else:
                        return d_v_d_alpha(x_fff=payoff_ffff, alpha_fff=alpha_ffff), 0
                if select_param_ffff == 'lambda':
                    return d_v_d_lambda(x_fff=payoff_ffff, beta_fff=beta_ffff), 0
                if select_param_ffff == 'beta':
                    return d_v_d_beta(x_fff=payoff_ffff, beta_fff=beta_ffff, lambda_fff=lambda_ffff), 0
                if select_param_ffff == 'gamma':
                    return 0, d_pi_d_c(p_fff=prob_ffff, c_fff=gamma_ffff) if (
                                payoff_ffff >= 0) else 0  # f payoff is < 0, then delta is used and gamma doesn't factor in
                if select_param_ffff == 'delta':
                    return 0, d_pi_d_c(p_fff=prob_ffff, c_fff=delta_ffff) if (
                                payoff_ffff < 0) else 0  # f payoff is >= 0, then gamma is used and delta doesn't factor in
                if select_param_ffff == 'phi':
                    return 0, 0

            def d_Voutcome_d_x(v_ffff, pi_ffff, d_v_d_x_ffff, d_pi_d_x_ffff):
                # d(V) / d(param) = d(v * pi) = [d(v)/d(param) * pi] + [v * d(pi)/d(param)]
                return (d_v_d_x_ffff * pi_ffff) + (v_ffff * d_pi_d_x_ffff)

            def calc_d_Pr_d_x(options_fff, j_chosen_fff, d_param_fff, alpha_fff, lambda_fff, beta_fff, gamma_fff,
                              delta_fff, phi_fff):
                def d_V_dx(outcomes_ffff, d_param_ffff, alpha_ffff, lambda_ffff, beta_ffff, gamma_ffff, delta_ffff):
                    t_d_Voutcome_d_x = list()
                    for _, t_outcome in outcomes_ffff.items():
                        t_pi = calc_outcome_pi(t_outcome['prob'], gamma_ffff if t_outcome['payoff'] > 0 else delta_ffff)
                        t_v = calc_outcome_v(t_outcome['payoff'], alpha_ffff, lambda_ffff, beta_ffff)
                        t_d_v_d_x, t_d_pi_d_x = d_Vcomponent_d_x(prob_ffff=t_outcome['prob'],
                                                                 payoff_ffff=t_outcome['payoff'],
                                                                 select_param_ffff=d_param_ffff,
                                                                 alpha_ffff=alpha_ffff, lambda_ffff=lambda_ffff,
                                                                 beta_ffff=beta_ffff,
                                                                 gamma_ffff=gamma_ffff, delta_ffff=delta_ffff)
                        t_d_Voutcome_d_x.append(d_Voutcome_d_x(t_v, t_pi, t_d_v_d_x, t_d_pi_d_x))
                        del t_pi, t_v, t_d_v_d_x, t_d_pi_d_x
                    return sum(t_d_Voutcome_d_x)

                # calc components specific to parameter
                t_d_V_dx_options = [
                    d_V_dx(j_opt.outcomes, d_param_fff, alpha_fff, lambda_fff, beta_fff, gamma_fff, delta_fff)
                    for j_opt in options_fff]

                # calc components common across derivatives
                t_option_vals = [j_opt.calc_option_value(alpha_fff, lambda_fff, beta_fff, gamma_fff, delta_fff) for j_opt in options_fff]
                t_exp_option_vals = [calc_exp_option_value(j_opt_val, phi_fff) for j_opt_val in t_option_vals]

                if d_param_fff != 'phi':
                    f_f = t_exp_option_vals[j_chosen_fff]
                    g_f = sum(t_exp_option_vals)
                    f_prime_f = phi_fff * t_exp_option_vals[j_chosen_fff] * t_d_V_dx_options[j_chosen_fff]
                    g_prime_f = phi_fff * np.dot(t_exp_option_vals, t_d_V_dx_options)
                    return (f_prime_f * g_f - g_prime_f * f_f) / (g_f ** 2)
                else:
                    f_f = t_exp_option_vals[j_chosen_fff]
                    g_f = sum(t_exp_option_vals)
                    f_prime_f = t_exp_option_vals[j_chosen_fff] * t_option_vals[j_chosen_fff]
                    g_prime_f = np.dot(t_exp_option_vals, t_option_vals)
                    return (f_prime_f * g_f - g_prime_f * f_f) / (g_f ** 2)

            if self.constrain_beta:
                grad_params_f = ['alpha', 'lambda', 'gamma', 'delta', 'phi']   #### remove 'beta' between 'lambda' and 'gamma'
            else:
                grad_params_f = ['alpha', 'lambda', 'beta', 'gamma', 'delta', 'phi']
            grad_values_f = list()
            for t_select_param in grad_params_f:
                grad_values_f.append(
                    (1 / prob_choice_ff) *
                    calc_d_Pr_d_x(options_fff=options_ff, j_chosen_fff=j_chosen_ff, d_param_fff=t_select_param,
                                  alpha_fff=alpha_ff, lambda_fff=lambda_ff, beta_fff=beta_ff,
                                  gamma_fff=gamma_ff, delta_fff=delta_ff, phi_fff=phi_ff)
                )

            return dict(zip(grad_params_f, grad_values_f))

        # ensure that the values stores for options, probs, choice_probs are update for current value of likelihood
        self.loglikeobs(params)  # may be able to remove this to halve the amount of execution time

        jacob_f = list()
        for cs_f in self.choice_situations_f:
            t_grad = calc_grad(options_ff=cs_f.options, j_chosen_ff=cs_f.get_choice_location(cs_f.choice),
                               prob_choice_ff=cs_f.prob_choice,
                               alpha_ff=self.alpha_f, lambda_ff=self.lambda_f, beta_ff=self.beta_f,
                               gamma_ff=self.gamma_f, delta_ff=self.delta_f,
                               phi_ff=self.phi_f)
            jacob_f.append([t_grad[t_key] for t_key in ['alpha', 'lambda', 'beta', 'gamma', 'delta', 'phi'] if t_key in list(t_grad.keys())])  # robust to dropping parameters (constraining)
        return np.array(jacob_f)


def generate_synthetic_data(n_hands, params_actual):
    choice_situations = list()
    for n in range(0, n_hands):
        t1_win_prob = random.random()
        t2_win_prob = random.random()

        def draw_payoff():
            return random.random() * 200 - 100

        t_choice_options = [Option(name='play', outcomes={'win': {'payoff': draw_payoff(), 'prob': t1_win_prob},
                                                          'lose': {'payoff': draw_payoff(), 'prob': 1 - t1_win_prob}}),
                            Option(name='fold', outcomes={'win': {'payoff': draw_payoff(), 'prob': t2_win_prob},
                                                          'lose': {'payoff': draw_payoff(), 'prob': 1 - t2_win_prob}})]
        t_choice_situation = ChoiceSituation(sit_options=t_choice_options[:])
        t_choice_situation.values = t_choice_situation.get_option_values(alpha_f=params_actual['alpha'],
                                                                         lambda_f=params_actual['lambda'],
                                                                         beta_f=params_actual['beta'],
                                                                         gamma_f=params_actual['gamma'],
                                                                         delta_f=params_actual['delta'])
        t_choice_situation.prob_options = t_choice_situation.get_option_probs(phi_f=params_actual['phi'])
        t_choice_situation.choice = (t_choice_situation.option_names[0] if random.random() < t_choice_situation.get_prob_choice(t_choice_situation.option_names[0]) else t_choice_situation.option_names[1])
        choice_situations.append(t_choice_situation)
        del t_choice_situation
    del n

    return choice_situations


def generate_choice_situations(player_f, game_hand_index_f, df_f=None):
    choice_situations_f = list()

    # calculate odds of winning select hands given starting hole cards

    for game_num, hands in game_hand_index_f.items():
        for hand_num in hands:
            # print('game %s hand %s' % (game_num, hand_num))
            big_blind = 100
            small_blind = 50
            payoff_units_f = 1

            tplay_win_prob = player_f.odds[game_num][hand_num]['slansky_prob']
            tplay_win_payoff = exp_win_seat_dict[str(player_f.seat_numbers[game_num][hand_num])]/payoff_units_f    # based on summary of data set, seat, slansky rank
            tplay_lose_payoff = exp_loss_seat_dict[str(player_f.seat_numbers[game_num][hand_num])]/payoff_units_f  # based on summary of data set, seat, slansky rank

            tfold_win_prob = 0  # cannot win under folding scenario
            if player_f.blinds[game_num][hand_num]['big']:
                tfold_lose_payoff = (big_blind * -1)/payoff_units_f
            elif player_f.blinds[game_num][hand_num]['small']:
                tfold_lose_payoff = (small_blind * -1)/payoff_units_f
            else:
                tfold_lose_payoff = 0/payoff_units_f

            t_choice_options = [Option(name='play', outcomes={'win': {'payoff': tplay_win_payoff, 'prob': tplay_win_prob},
                                                              'lose': {'payoff': tplay_lose_payoff, 'prob': 1 - tplay_win_prob}}),
                                Option(name='fold', outcomes={'lose': {'payoff': tfold_lose_payoff, 'prob': 1 - tfold_win_prob}})]  # 'win': {'payoff': draw_payoff(), 'prob': tfold_win_prob}

            if df_f is not None:
                t_post_loss_bool = df_f.loc[(df_f.player == player_f.name) &
                                          (df_f.game == float(game_num)) &
                                          (df_f.hand == float(hand_num)), 'prev_outcome_loss'].bool()
            else:
                t_post_loss_bool = None

            t_choice_situation = ChoiceSituation(sit_options=t_choice_options[:],
                                                 sit_choice="fold" if player_f.actions[game_num][hand_num]['preflop'] == 'f' else "play",
                                                 slansky_strength=player_f.odds[game_num][hand_num]['slansky'],
                                                 stack_rank=player_f.stack_ranks[game_num][hand_num],
                                                 post_loss=t_post_loss_bool)

            choice_situations_f.append(t_choice_situation)

            del t_choice_situation
    del game_num, hands, hand_num
    return choice_situations_f


def filter_choice_situations(choice_situations_f, select_slansky_ranks_f=None, select_stack_ranks_f=None, select_type_loss_data_f=None):
    ind_to_keep = list()

    for t_ind_cs in range(0, len(choice_situations_f)):
        t_keep_bool = True

        # keep only select slansky ranks
        if select_slansky_ranks_f is not None:
            if choice_situations_f[t_ind_cs].slansky_strength not in select_slansky_ranks_f:
                t_keep_bool = False

        # keep only select stack ranks
        if select_stack_ranks_f is not None:
            if choice_situations_f[t_ind_cs].stack_rank not in select_stack_ranks_f:
                t_keep_bool = False

        # keep only select hands based on following loss or not
        if select_type_loss_data_f is not None:
            if (select_type_loss_data_f == 'post_loss') and (not choice_situations_f[t_ind_cs].post_loss):
                t_keep_bool = False
            if (select_type_loss_data_f == 'not_post_loss') and choice_situations_f[t_ind_cs].post_loss:
                t_keep_bool = False

        # pop element if filtered out on any condition
        if t_keep_bool:
            ind_to_keep.append(t_ind_cs)

    return [copy.deepcopy(choice_situations_f[i]) for i in ind_to_keep]


# for examining data characteristics
def get_play_perc(choice_situations_f, select_slansky_ranks_f, select_stack_ranks_f):
    t_play_count = 0
    t_cond_sit_count = 0
    for cs in choice_situations_f:
        if (cs.slansky_strength in select_slansky_ranks_f) and (cs.stack_rank in select_stack_ranks_f):
            t_cond_sit_count += 1
            if cs.choice == 'play':
                t_play_count += 1
    return t_play_count/t_cond_sit_count, t_cond_sit_count


# ------------- SET PARAMETERS ------------------------------
param_names_actual = ['alpha', 'lambda', 'beta', 'gamma', 'delta', 'phi']
success_event_name = 'fold'
fail_event_name = 'play'
select_stack_ranks = [1, 2, 3, 4, 5, 6, 7] # works for [3, 4]; [1, 2] and [5, 6] and [1, 2, 3, 4, 5, 6]  does not have interior solution for gamma
select_slansky_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9] # works for [1, 2, 8, 9]
select_type_loss_data = ['all']     # 'post_loss', 'not_post_loss',
constrain_beta_TF = True

# ------------- GENERATE SYNTHETIC DATA ----------------------
# param_values_actual = [0.88, 2.25, 0.88, 0.61, 0.69, 0.1]
# params_actual = dict(zip(param_names_actual, param_values_actual))
# n_hands = 3000
# choice_situations = generate_synthetic_data(n_hands, params_actual, success_event_name, fail_event_name)    # t_choice_options = [Option(name='play', outcomes={'win': {'payoff': 100, 'prob': 0.6}, 'lose': {'payoff': -100, 'prob': 0.4}}), Option(name='fold', outcomes={'win': {'payoff': 0.01, 'prob': 0.5}, 'lose': {'payoff': -10, 'prob': 0.5}})]
#
# select_players = ['Pluribus'] # Not used, only for matching structure of real data loops
# select_player_comps = {'Pluribus': 'Pluribus'}  # Not used, only for matching structure of real data loops

# -------------- PROCESS ACTUAL DATA ----------------------------
with open("python_hand_data.pickle", 'rb') as f:
    data = pickle.load(f)
players = data['players']
games = data['games']
df = data['df']

# select_players = [p.name for p in players]   #### naive, should be selected by hand
# select_player_comps = dict(zip(select_players, select_players))   #### naive setting, should be done intelligently and put in parameters section

select_players = ['Budd']
select_player_comps = {'Budd': 'Budd'}


# --- configure data and run estimation
def config_data(select_player, select_player_comps, select_type_loss_data, select_slansky_ranks_f, select_stack_ranks_f):
    # ----- comment out for synthetic fitting -------
    # create player-specific game, hand index
    print('Creating data sets for %s' % select_player)
    choice_situations = list()
    for t_player in [i for i in range(0, len(players)) if players[i].name in select_player_comps[select_player]]:  # , 'Gogo', 'MrWhite', 'Hattori']]:
        game_hand_player_index = create_game_hand_index(players[t_player])  # 11 is Budd, strong coef from linear regression
        t_choice_situations = generate_choice_situations(players[t_player], game_hand_player_index, df)  # select_slansky_ranks_ff=select_slansky_ranks_f, select_stack_ranks_ff=select_stack_ranks_f)
        choice_situations = choice_situations + t_choice_situations
    del t_choice_situations

    print("%d choice situations processed" % len(choice_situations))

    # filter out slansky ranks, stack ranks, and data split for post loss vs. not
    choice_situations_dict = dict()
    for t_select_type_loss_data in select_type_loss_data:
        choice_situations_dict.update({t_select_type_loss_data: filter_choice_situations(choice_situations,
                                                                                         select_slansky_ranks_f=select_slansky_ranks_f,
                                                                                         select_stack_ranks_f=select_stack_ranks_f,
                                                                                         select_type_loss_data_f=t_select_type_loss_data)})
        print("%d choice situations included for %s type of data" % (len(choice_situations_dict[t_select_type_loss_data]), t_select_type_loss_data))
        print("%3.3f choose play for %d total situations" % get_play_perc(choice_situations_dict[t_select_type_loss_data], select_slansky_ranks_f, select_stack_ranks_f))
    return choice_situations_dict


def estimate_model(choice_situations_dict, select_type_loss_data, constrain_beta_TF):
    # ------------ FIT MODEL -------------------
    # Optimization parameters
    maxiter = 100
    pgtol = 1e-6
    ftol = 1e-8
    if constrain_beta_TF:
        start_params = [0.5, 1.75, 0.6, 0.6, 0.2]  ### removed 0.5 as beta starting point
        bounds_params = [(1e-2, 1 - 1e-2), (1e-2, 3), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2)]  ### removed (1e-2, 1 - 1e-2), as beta bounds
    else:
        start_params = [0.5, 1.75, 0.5, 0.6, 0.6, 0.2]
        bounds_params = [(1e-2, 1 - 1e-2), (1e-2, 3), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2)]

    # --- Fit model
    model_results = dict()
    models = dict()
    for type_loss_data in select_type_loss_data:
        print('\n\n\n\nEstimating %s model' % type_loss_data)
        model = ProspectModel(endog=np.array([1 if x.choice == success_event_name else 0 for x in choice_situations_dict[type_loss_data]]),
                              choice_situations_f=choice_situations_dict[type_loss_data],
                              constrain_beta=constrain_beta_TF)  # add arg: phi_f=phi to take out phi in deriv

        model_result = model.fit(start_params=start_params, method='LBFGS', maxiter=maxiter, disp=True,
                                 bounds=bounds_params, pgtol=pgtol, factr=ftol)
        print(model_result.summary())
        try:
            print('Actual parameters: %s' % params_actual)
        except:
            pass
        print('Gradient at solution: %s' % dict(zip(param_names_actual, model.score(model_result.params))))
        model_results.update({type_loss_data: model_result})
        models.update({type_loss_data: model})

    return model_results, models

player_model_results = dict()
player_models = dict()
player_choice_situations = dict()
for t_select_player in select_players:
    # get choice situaitons
    t_choice_situations_dict = config_data(t_select_player, select_player_comps, select_type_loss_data, select_slansky_ranks, select_stack_ranks)
    player_choice_situations.update({t_select_player: t_choice_situations_dict})

    # estimate model
    # t_estimation_output, t_model = estimate_model(t_choice_situations_dict, select_type_loss_data, constrain_beta_TF)    # choice_situations, choice_situations_post_loss, choice_situations_not_post_loss
    # player_model_results.update({t_select_player: t_estimation_output})
    # player_models.update({t_select_player: t_model})

    # del t_estimation_output, t_model, t_choice_situations_dict


# ------ Examine player descriptive stats; comment out or delete later -----------
choice_situations = player_choice_situations['Budd']['all']
df_play_perc = pd.DataFrame(index=select_slansky_ranks, columns=select_stack_ranks)
df_num_obs = pd.DataFrame(index=select_slansky_ranks, columns=select_stack_ranks)
for t_slansky_rank in select_slansky_ranks:
    for t_stack_rank in select_stack_ranks:
        t_slansky_rank = ([t_slansky_rank] if (type(t_slansky_rank) is not list) else t_slansky_rank)
        t_stack_rank = ([t_stack_rank] if (type(t_stack_rank) is not list) else t_stack_rank)
        t_perc, t_num = get_play_perc(choice_situations, t_slansky_rank, t_stack_rank)
        df_play_perc.loc[t_slansky_rank, t_stack_rank] = round(t_perc*1e2, 1)
        df_num_obs.loc[t_slansky_rank, t_stack_rank] = t_num
print(df_play_perc)
print(df_num_obs)

for j in range(1, 8):
    print(j, get_play_perc(choice_situations, [i for i in range(3, 10)], [j]))

t_slansky = [1, 2, 3, 4, 7, 8, 9]    #[i for i in range(3, 10)]
t_stacks = [2, 3]
t = filter_choice_situations(player_choice_situations['Budd']['all'], select_slansky_ranks_f=t_slansky, select_stack_ranks_f=t_stacks, select_type_loss_data_f='all')
get_play_perc(t, t_slansky, t_stacks)
estimate_model({'all': t}, select_type_loss_data, constrain_beta_TF)

# ---------------------------- END DELETE SECTION --------------------------------


# create dataframe(s) for examination
# import pandas as pd
#
# param_locs = dict(zip(['alpha', 'lambda', 'gamma', 'delta'], [0, 1, 2, 3]))
#
# df_model_results_dict = dict()
# for t_param, t_loc in param_locs.items():
#     t_df = pd.DataFrame(columns=['all', 'post_loss', 'not_post_loss'])
#     for t_player in player_model_results.keys():
#         for t_type, t_model_result in player_model_results[t_player].items():
#             t_df.loc[t_player, t_type] = t_model_result.params[t_loc]
#     df_model_results_dict.update({t_param: t_df})
#     del t_df
#
# for k, v in df_model_results_dict.items():
#     print('\n')
#     print('Parameter %s' % k)
#     print(v)

# with open('model_results.pickle', 'wb') as f:
#     pickle.dump(df_model_results_dict, f)

# =============== Graphical examination ====================
# Plotting parameters
select_param_name1 = 'lambda'
select_param_name2 = 'phi'
param_names_fit = ['alpha', 'lambda', 'gamma', 'delta', 'phi']  ### removed beta
calc_contour_info = True
n_1D_mesh_points = 10
plot_param_bounds = [(1e-2, 1 - 1e-2), (1e-2, 3), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2)]    ### removed beta bounds

select_model_result = player_model_results['Pluribus']['all']
select_model = player_models['Pluribus']['all']
print('WARNING: number of params in names does not match number in values') if len(param_names_fit) != len(select_model_result.params) else None


def get_result_info(select_model_f, select_param_name_f, param_names_f, param_values_f, param_domain_f, calc_contour_info_f=False, select_param_name2_f=None, n_1D_mesh_points=10):
    p_loc_in_params = param_names_f.index(select_param_name_f)
    p_vec = np.linspace(param_domain_f[select_param_name_f][0], param_domain_f[select_param_name_f][1], 25)

    LL_p = list()
    score_p = list()
    for p in p_vec[1:]:
        t_params = param_values_f[0:p_loc_in_params] + [p] + param_values_f[(p_loc_in_params + 1):]
        LL_p.append(np.sum(select_model_f.loglikeobs(t_params)))
        score_p.append(np.sum(select_model_f.score_obs(np.array(t_params))[:, p_loc_in_params]))
    finite_diff_deriv_LL_p = [(LL_p[i_f] - LL_p[i_f - 1]) / (p_vec[i_f] - p_vec[i_f - 1]) for i_f in range(1, len(LL_p))]

    p_vec1 = None
    p_vec2 = None
    LL_matrix = None
    if calc_contour_info_f and (select_param_name2_f is not None):
        # ----- contour plot
        p_vec1 = np.linspace(param_domain_f[select_param_name_f][0], param_domain_f[select_param_name_f][1], n_1D_mesh_points)
        p_vec2 = np.linspace(param_domain_f[select_param_name2_f][0], param_domain_f[select_param_name2_f][1], n_1D_mesh_points)
        LL_matrix = np.empty(shape=[len(p_vec1), len(p_vec2)])
        p_loc1 = param_names_f.index(select_param_name_f)
        p_loc2 = param_names_f.index(select_param_name2_f)

        for i in range(0, len(p_vec1)):
            for j in range(0, len(p_vec2)):
                t_base_params = param_values_f.copy()
                t_base_params[p_loc1] = p_vec1[i]
                t_base_params[p_loc2] = p_vec2[j]
                LL_matrix[i, j] = np.sum(select_model_f.loglikeobs(t_base_params))

    return p_vec, LL_p, score_p, finite_diff_deriv_LL_p, {select_param_name_f: p_vec1}, {select_param_name2_f: p_vec2}, LL_matrix


def plot_functions(select_param_name_f, p_vec_f, LL_p_f, score_p_f=None, finite_diff_deriv_LL_p_f=None,
                   plot_LL_num_deriv=False, plot_num_analytical_deriv=False,
                   plot_contour=False, XX_f=None, YY_f=None, LL_matrix_f=None, n_levels_f=10):
    # --- Likelihood function and derivatives
    if plot_LL_num_deriv:
        # ---- LL function and numerical derivative
        fig3, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel(select_param_name_f)
        ax1.set_ylabel('LL', color=color)
        ax1.plot(p_vec_f[1:], LL_p_f, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('numerical derivative approximation', color=color)  # we already handled the x-label with ax1
        if finite_diff_deriv_LL_vec is not None:
            ax2.plot(p_vec_f[1:], finite_diff_deriv_LL_p_f[0:] + [None], color=color)
        if score_p_f is not None:
            ax2.plot(p_vec_f[1:], score_p_f, color='green')
        ax2.tick_params(axis='y', labelcolor=color)

        fig3.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    # --- Derivatives only
    if plot_num_analytical_deriv:
        fig1, ax1 = plt.subplots()
        ax1.plot(p_vec_f[1:], score_p_f, color='green')
        ax1.plot(p_vec_f[1:], [None] + finite_diff_deriv_LL_p_f, color='blue')
        ax1.set_xlabel(select_param_name_f)
        ax1.set_ylabel('Value')
        ax1.legend(['analytical', 'numerical'])
        plt.show()

        # ---- for different axes scales -----
        # fig2, ax1 = plt.subplots()
        #
        # color = 'tab:green'
        # ax1.set_xlabel(select_param_name_f)
        # ax1.set_ylabel('analytical derivative', color=color)
        # ax1.plot(p_vec_f[1:], score_p_f, color=color)
        # ax1.tick_params(axis='y', labelcolor=color)
        #
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #
        # color = 'tab:blue'
        # ax2.set_ylabel('numerical derivative approximation', color=color)  # we already handled the x-label with ax1
        # ax2.plot(p_vec_f[1:], finite_diff_deriv_LL_p_f[0:] + [None], color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        #
        # fig2.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()

    # --- Contour plot
    if plot_contour and (list(XX_f.values())[0] is not None) and (list(YY_f.values())[0] is not None):
        fig, ax1 = plt.subplots()
        CS = plt.contour(list(XX_f.values())[0], list(YY_f.values())[0], LL_matrix_f, n_levels_f)
        ax1.set_xlabel(list(XX_f.keys())[0])
        ax1.set_ylabel(list(YY_f.keys())[0])
        ax1.set_title('LL contour plot')
        cbar1 = fig.colorbar(CS)
        cbar1.ax.set_ylabel('LL')
        cbar1.add_lines(CS)
        ax1.plot(list(XX_f.values())[0][np.where(LL_matrix_f == np.nanmax(LL_matrix_f))[0]],
                 list(YY_f.values())[0][np.where(LL_matrix_f == np.nanmax(LL_matrix_f))[1]],
                 marker="o")


p_vec, LL_vec, score_vec, finite_diff_deriv_LL_vec, XX, YY, LL_matrix = get_result_info(select_model_f=select_model, select_param_name_f=select_param_name1, param_names_f=param_names_fit, param_values_f=list(select_model_result.params),
                                                                                        param_domain_f=dict(zip(param_names_fit, plot_param_bounds)),
                                                                                        calc_contour_info_f=calc_contour_info, select_param_name2_f=select_param_name2, n_1D_mesh_points=n_1D_mesh_points)

plot_functions(select_param_name1, p_vec, LL_vec, score_vec, finite_diff_deriv_LL_vec, plot_LL_num_deriv=True, plot_num_analytical_deriv=True, plot_contour=True, XX_f=XX, YY_f=YY, LL_matrix_f=LL_matrix)

# ---- AVP plots -----
choice_actual = list()
choice_pred = list()
for cs in choice_situations:
    # cs.set_model_values(model_result.params[0], model_result.params[1], model_result.params[2], model_result.params[3], model_result.params[4], phi)    ### artificially fills in beta when alpha=beta constraint
    cs.predict_choice()
    choice_actual.append(cs.choice)
    choice_pred.append(cs.pred_choice)
del cs

# ----- Confusion matrix -----
print(confusion_matrix(choice_actual, choice_pred, labels=['fold', 'play']))

#---------- ARCHIVE ----------------#
#--- investigate outcomes by player and hand strength ---#
# temp = df.groupby(['player', 'slansky_rank'])['preflop_fold'].apply(lambda x: x.value_counts()/x.count())
# t_colors = dict(zip(list(df.player.unique()), ['fuchsia', 'green', 'yellow', 'blue', 'teal', 'orange', 'deeppink', 'black', 'mediumslateblue', 'orangered', 'saddlebrown', 'rebeccapurple', 'maroon', 'olive']))
# for s in df.slansky_rank.unique():
#     for p in df.player.unique():
#         try:
#             plt.plot(s, temp.loc[p][s][True], 'o', color=t_colors[p])
#         except KeyError:
#             pass
# plt.legend(t_colors)


#------ Test quality of data by generating choice according to parameters -----#
# for cs in choice_situations:
#     # cs.set_model_values(0.2, 1, 0.2, 0.6, 0.6, phi)  ### artificially fills in beta when alpha=beta constraint
#     # cs.set_model_values(0.88, 2.25, 0.88, 0.61, 0.69, phi)    ### artificially fills in beta when alpha=beta constraint
#     # cs.set_model_values(param_values_actual[0], param_values_actual[1], param_values_actual[2], param_values_actual[3], param_values_actual[4], phi)  ### artificially fills in beta when alpha=beta constraint
#     cs.set_model_values(model_result.params[0], model_result.params[1], model_result.params[0], model_result.params[2], model_result.params[3], phi)  ### artificially fills in beta when alpha=beta constraint
#
#     cs.predict_choice()
#     # cs.choice = (cs.option_names[0] if random.random() < cs.get_prob_choice(cs.option_names[0]) else cs.option_names[1])
#
# probs_play = [cs.prob_options[0] for cs in choice_situations]
# plt.hist([p for p in probs_play])
#
# evs_play = [cs.options[0].outcomes['win']['payoff'] * cs.options[0].outcomes['win']['prob'] + cs.options[0].outcomes['lose']['payoff'] * cs.options[0].outcomes['lose']['prob'] for cs in choice_situations]
# evs_fold = [cs.options[1].outcomes['lose']['payoff'] * cs.options[1].outcomes['lose']['prob'] for cs in choice_situations]
# plt.hist(evs_play)
# plt.hist(evs_fold)

# vals_play = [cs.values[0] - cs.values[1] for cs in choice_situations]
# plt.hist(vals_play)

# choice_situations = [choice_situations[i] for i in range(0, len(choice_situations)) if (choice_situations[i].options[0].outcomes['win']['payoff'] * choice_situations[i].options[0].outcomes['win']['prob'] +
#                                                                                         choice_situations[i].options[0].outcomes['lose']['payoff'] * choice_situations[i].options[0].outcomes['lose']['prob']) > -100]
# choice_situations = [cs for cs in choice_situations if ((cs.pred_choice == 'play') or ((cs.pred_choice == 'fold') and (random.random() < 0.15)))]
# choice_situations = [cs for cs in choice_situations if ((cs.choice == 'play') or ((cs.choice == 'fold') and (random.random() < 0.3)))]
