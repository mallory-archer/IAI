import pickle
import os
import copy
import numpy as np
from scipy.optimize import minimize, Bounds, fsolve
import binascii

import matplotlib.pyplot as plt

from classify_gambles import identify_omega_range
from classify_gambles import classify_gamble_types


class Option:
    def __init__(self, name, outcomes):
        self.name = name
        self.outcomes = outcomes


class ChoiceSituation:
    def __init__(self, sit_options, sit_choice=None, slansky_strength=None, stack_rank=None, seat=None, post_loss=None, post_win=None, post_neutral=None, post_loss_excl_blind_only=None, post_win_excl_blind_only=None, post_neutral_or_blind_only=None, tags=None):
        self.options = sit_options
        self.option_names = [x.name for x in sit_options]
        self.choice = sit_choice
        self.slansky_strength = slansky_strength
        self.stack_rank = stack_rank
        self.seat = seat
        self.post_loss = post_loss
        self.post_win = post_win
        self.post_neutral = post_neutral
        self.post_loss_excl_blind_only = post_loss_excl_blind_only
        self.post_win_excl_blind_only = post_win_excl_blind_only
        self.post_neutral_or_blind_only = post_neutral_or_blind_only
        self.CRRA_ordered_gamble_info = None
        self.CRRA_ordered_gamble_type = None
        self.omega_indifference = None
        self.tags = tags

    def get_gamble_info(self, lambda_list_f=[0.5, 1.5, 10], omega_list_f=[x / 100 for x in range(-500, 50000, 10)]):
        t_omega_range_dict = identify_omega_range(payoff_win_f=[x.outcomes['win']['payoff'] for x in self.options if x.name == 'play'][0],
                                                  payoff_lose_f=[x.outcomes['lose']['payoff'] for x in self.options if x.name == 'play'][0],
                                                  payoff_fold_f=[x.outcomes['lose']['payoff'] for x in self.options if x.name == 'fold'][0],
                                                  prob_win_f=[x.outcomes['win']['prob'] for x in self.options if x.name == 'play'][0],
                                                  lambda_list_f=lambda_list_f,
                                                  omega_list_f=omega_list_f)  # note: this omega range may need to be changed. If not large enough will throw error and terminate

        for l_level, l_dict in t_omega_range_dict.items():
            l_dict.update({'gamble_type': classify_gamble_types(l_dict['prob_util']['play_util'],
                                                                l_dict['prob_util']['fold_util'])})
            if l_dict['gamble_type'] == 'prob_risk_decreases_omega_increases':
                t_vec = [x - l_dict['prob_util']['prob'][i - 1] for i, x in enumerate(l_dict['prob_util']['prob'])]
                if not ((not all([y > 0 for y in t_vec])) and (not all([y < 0 for y in t_vec])) and (any([y > 0.5 for y in l_dict['prob_util']['prob']]))):
                    l_dict.update({'classification_warning': True})
                    print('WARNING: game %s hand %s lambda=%3.2f classified as %s but some conditions are not met for this classification' % (
                            self.tags['game'],
                            self.tags['hand'],
                            l_level, 'prob_risk_decreases_omega_increases'))
                else:
                    l_dict.update({'classification_warning': False})
            elif l_dict['gamble_type'] in ['ERROR', 'prob_risk_increases_omega_increases']:
                l_dict.update({'classification_warning': True})
                print('CHECK INVALID CLASSIFICATION: %s for game %s hand %s, lambda=%f' % (l_dict['gamble_type'], self.tags['game'], self.tags['hand'], l_level))
            else:
                l_dict.update({'classification_warning': False})

        return t_omega_range_dict

    def evaluate_gamble_info(self):
        if self.CRRA_ordered_gamble_info is None:
            print("Gamble info not calculated. Call 'get_gamble_info' method first")
        else:
            if all([x['gamble_type'] == 'risky_dominant' for x in self.CRRA_ordered_gamble_info.values()]):
                self.CRRA_ordered_gamble_type = 'risky_dominant'
            elif all([x['gamble_type'] == 'safe_dominant' for x in self.CRRA_ordered_gamble_info.values()]):
                self.CRRA_ordered_gamble_type = 'safe_dominant'
            elif any([x['gamble_type'] == 'prob_risk_decreases_omega_increases' for x in self.CRRA_ordered_gamble_info.values()]):
                self.CRRA_ordered_gamble_type = 'prob_risk_decreases_omega_increases'
            else:
                'CHECK GAMBLE TYPES IN CRRA_ordered_gamble_info'

    def find_omega_indifference(self):
        if self.CRRA_ordered_gamble_type is None:
            print("CRRA_ordered_gamble_type is None, call 'evalute_gamble_info' method first")
        else:
            try:
                if self.CRRA_ordered_gamble_type == 'prob_risk_decreases_omega_increases':
                    self.omega_indifference = fsolve(lambda x: calc_CRRA_utility(outcomes=[x.outcomes for x in self.options if x.name == 'play'][0].values(), omega=x) - calc_CRRA_utility(outcomes=[x.outcomes for x in self.options if x.name == 'fold'][0].values(), omega=x), x0 = 0.4)[0]
            except:
                print('Could not find indifference point for game %s hand %s player %s and gamble type %s' % (self.tags['game'], self.tags['hand'], self.tags['player'], self.CRRA_ordered_gamble_type))

    def plot_single_prob_line(self, t_data):
        t_omega_range = [x / 10000 for x in range(int(t_data['min_omega'] * 10000), int(t_data['max_omega'] * 10000),
                                                  int((t_data['max_omega'] - t_data['min_omega']) / (
                                                      len(t_data['prob_util']['prob'])) * 10000))]
        plt.plot(t_omega_range[0:len(t_data['prob_util']['prob'])], t_data['prob_util']['prob'])
        plt.xlabel('omega')
        plt.ylabel('probability of choosing risker gamble, play')

    def plot_prob(self, add_labels=True):
        for l, t_data in self.CRRA_ordered_gamble_info.items():
            t_vec = [x - t_data['prob_util']['prob'][i - 1] for i, x in enumerate(t_data['prob_util']['prob'])]
            if (t_data['gamble_type'] == 'prob_risk_decreases_omega_increases') and (not ((not all([y > 0 for y in t_vec])) and (not all([y < 0 for y in t_vec])) and (any([y > 0.5 for y in t_data['prob_util']['prob']])))):
                print('WARNING: game %s hand %s classified as %s but some conditions are not met for this classification' %
                      (self.tags['game'], self.tags['hand'], t_data['gamble_type']))
            try:
                self.plot_single_prob_line(t_data)
            except ZeroDivisionError:
                print('check calc of t_omega range for lambda = %s gamble_type = %s' % (l, t_data['gamble_type']))
            del t_vec
        if add_labels:
            plt.title('Gamble classified as %s\ngame %s hand %s player %s' % (self.CRRA_ordered_gamble_type, self.tags['game'], self.tags['hand'], self.tags['player']))
            plt.legend(['lambda=%3.2f' % x for x in self.CRRA_ordered_gamble_info.keys()])
        del l, t_data

    def print(self):
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))


class RandomUtilityModel:
    def __init__(self, data_f, param_names_f=['kappa', 'lambda', 'omega'], LL_form_f='RUM'):
        self.data_f = data_f
        self.kappa_f = None
        self.lambda_f = None
        self.omega_f = None
        self.results = None

        self.param_names_f = param_names_f  # ['kappa', 'lambda', 'omega']  ###################
        self.LL_form = LL_form_f
        self.init_params = [0.5] * len(param_names_f)
        self.lb={'kappa': 0.000, 'lambda': 0.0000, 'omega': 0}
        self.ub={'kappa': 0.25, 'lambda': 100, 'omega': 2}
        self.est_params = {'method': 'l-bfgs-b',
                           'bounds': self.create_bounds_object(),
                           'tol': 1e-12,
                           'options': {'disp': False, 'maxiter': 500}}

    def create_bounds_object(self):
        return Bounds(lb=[self.lb[self.param_names_f[i]] for i in range(len(self.param_names_f))],
                      ub=[self.ub[self.param_names_f[i]] for i in range(len(self.param_names_f))])

    def negLL_RUM(self, params):
        def calc_LLi(X, Y, I, util_X, util_Y, kappa, lam):
            return (X + I / 2) * np.log(calc_RUM_prob(util_i=util_X, util_j=[util_X, util_Y], lambda_f=lam, kappa_f=kappa)) + \
                   (Y + I / 2) * np.log(calc_RUM_prob(util_Y, [util_X, util_Y], lam, kappa))

        for att in self.param_names_f:
            setattr(self, att + '_f', params[self.param_names_f.index(att)])
        # self.kappa_f = params[self.param_names_f.index('kappa')]
        # self.lambda_f = params[self.param_names_f.index('lambda')]
        # self.omega_f = params[self.param_names_f.index('omega')]

        LLi = list()
        t_total_obs = 0
        for rank in self.data_f.keys():
            for seat in self.data_f[rank].keys():
                for t_select_item in self.data_f[rank][seat]:
                    t_LLi = calc_LLi(X=t_select_item['n_chosen']['play'],
                                     Y=t_select_item['n_chosen']['fold'],
                                     I=0,
                                     util_X=calc_CRRA_utility(outcomes=t_select_item['params']['play'],
                                                              omega=self.omega_f),
                                     util_Y=calc_CRRA_utility(outcomes=t_select_item['params']['fold'],
                                                              omega=self.omega_f),
                                     kappa=self.kappa_f,
                                     lam=self.lambda_f)
                    LLi.append(t_LLi)
                    t_total_obs += sum(t_select_item['n_chosen'].values())
                del t_select_item
        return -sum(LLi)/t_total_obs

    def negLL_RPM(self, params):
        ### WARNING: DON'T HAVE CONFIGURED FOR Y STOCHASTICALLY DOMINATES INSTANCES - CHECK THIS IS CORRECT?
        def calc_LLi(X_mixed, Y_mixed, omega_xy, omega, lam, kappa, X_xdom, Y_xdom):
            return (X_mixed * np.log(calc_RPM_prob(util_i=omega_xy, util_j=[omega_xy, omega], lambda_f=lam, kappa_f=kappa))) + \
                   (Y_mixed * np.log(calc_RPM_prob(util_i=omega, util_j=[omega, omega_xy], lambda_f=lam, kappa_f=kappa))) + \
                   (X_xdom * np.log(1 - kappa)) + (Y_xdom * np.log(kappa))

        for att in self.param_names_f:
            setattr(self, att + '_f', params[self.param_names_f.index(att)])

        LLi = list()
        t_total_obs = 0
        for rank in self.data_f.keys():
            for seat in self.data_f[rank].keys():
                for t_select_item in self.data_f[rank][seat]:
                    if (t_select_item['CRRA_gamble_type'] == 'prob_risk_decreases_omega_increases') and (t_select_item['omega_indifference'] is not None):
                        t_wxy = t_select_item['omega_indifference']
                    else:
                        t_wxy = 0   #### check implications of setting to 0
                    t_Xmixed = t_select_item['n_chosen']['play'] if (t_select_item['CRRA_gamble_type'] == 'prob_risk_decreases_omega_increases') else 0
                    t_Ymixed = t_select_item['n_chosen']['fold'] if (t_select_item['CRRA_gamble_type'] == 'prob_risk_decreases_omega_increases') else 0
                    t_X_xdom = t_select_item['n_chosen']['play'] if (t_select_item['CRRA_gamble_type'] == 'risky_dominant') else 0
                    t_Y_xdom = t_select_item['n_chosen']['fold'] if (t_select_item['CRRA_gamble_type'] == 'risky_dominant') else 0
                    if (t_Xmixed + t_Ymixed + t_X_xdom + t_Y_xdom) != 1:
                        print('WARNING: CHECK CALCULATION OF TYPE COUNTS IN RPM NEG LL FOR RANK %s SEAT %s' % (rank, seat))
                    t_LLi = calc_LLi(X_mixed=t_Xmixed,
                                     Y_mixed=t_Ymixed,
                                     omega_xy=t_wxy,
                                     omega=self.omega_f,
                                     lam=self.lambda_f,
                                     kappa=self.kappa_f,
                                     X_xdom=t_X_xdom,
                                     Y_xdom=t_Y_xdom
                                     )
                    LLi.append(t_LLi)
                    t_total_obs += sum(t_select_item['n_chosen'].values())
                    del t_wxy, t_Xmixed, t_Ymixed, t_X_xdom, t_Y_xdom, t_LLi
                del t_select_item
            del seat
        del rank
        return -sum(LLi)/t_total_obs

    def set_est_params(self, est_param_dict):
        for k, v in est_param_dict.items():
            if (k == 'lb') or (k == 'ub'):
                self.__setattr__(k, v)
            else:
                self.est_params[k] = v

    def fit(self, init_params=None, LL_form=None, est_param_dict={}):
        if init_params is not None:
            self.init_params = init_params

        if LL_form is not None:
            self.LL_form = LL_form

        self.set_est_params(est_param_dict)
        self.est_params['bounds'] = self.create_bounds_object()

        if self.LL_form == 'RPM':
            t_LL = self.negLL_RPM
        elif self.LL_form == 'RUM':
            t_LL = self.negLL_RUM
        else:
            print('INVALID PROB MODEL - must choose RPM or RUM')

        self.results = minimize(fun=t_LL, x0=self.init_params, **self.est_params)

    def print(self):
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))


def generate_choice_situations(player_f, prob_dict_f, payoff_dict_f, payoff_shift_player_list_f=None):
    def get_payoff_dist_chars(payoff_dict_ff, player_ff=None):
        # get payoffs to examine min payoff for shifting into positive domain
        # if no player name is provided, then lowest payoff over all situations is found
        t_obs_avg_payoffs = list()
        if player_ff is None:
            t_player_list = payoff_dict_ff.keys()
        else:
            t_player_list = player_ff

        for t_player_name in t_player_list:
            for game_num, hands in payoff_dict_ff[t_player_name].items():
                for hand_num, payoffs in hands.items():
                    for outcome, outcomes in payoffs.items():
                        try:
                            t_obs_avg_payoffs.append(list(outcomes.values()))
                        except KeyError:
                            print('Error for keys game %s and hand %s' % (game_num, hand_num))
                    del outcome, outcomes
                del hand_num, payoffs
            del game_num, hands
        del t_player_name
        t_flat_list = [x for y in t_obs_avg_payoffs for x in y]

        # plt.figure()
        # plt.hist([x for y in t_obs_avg_payoffs for x in y])
        # plt.title('Histogram of predicted payoffs for %s from payoff dictionary (unadjusted)' % t_player_list)
        del t_player_list
        return {'min': min(t_flat_list), 'max': max(t_flat_list),
                'mean': np.mean(t_flat_list), 'stdev': np.std(t_flat_list)}

    choice_situations_f = list()
    num_choice_situations_dropped = 0

    big_blind = 100
    small_blind = 50
    t_payoff_dist_chars = get_payoff_dist_chars(payoff_dict_f, player_ff=payoff_shift_player_list_f)

    # payoff_units_f = 1/big_blind  # linear transformation
    # payoff_shift_f = get_min_observed_payoff(payoff_dict_f, player_ff=payoff_shift_player_list_f) * -1                # linear transformation
    payoff_shift_f = (((t_payoff_dist_chars['min'] - t_payoff_dist_chars['mean']) / t_payoff_dist_chars['stdev']) * -1) + 1     # standard normal transformatoin

    for game_num, hands in payoff_dict_f[player_f.name].items():
        for hand_num, probs in hands.items():
        #     if (hand_num == '166'):
        #         break
        # if (game_num == '42'):
        #     break

            # --- play
            tplay_win_prob = prob_dict_f[player_f.name][game_num][hand_num]
            tplay_win_payoff = payoff_dict_f[player_f.name][game_num][hand_num]['play']['win']
            tplay_lose_payoff = payoff_dict_f[player_f.name][game_num][hand_num]['play']['lose']

            # --- fold
            tfold_win_prob = 0  # cannot win under folding scenario
            tfold_lose_payoff = payoff_dict_f[player_f.name][game_num][hand_num]['fold']['lose']

            # --- shift/scale payoffs ----
            # tplay_win_payoff = (tplay_win_payoff + payoff_shift_f) * payoff_units_f   # linear translation
            # tplay_lose_payoff = (tplay_lose_payoff + payoff_shift_f) * payoff_units_f # linear translation
            # tfold_lose_payoff = (tfold_lose_payoff + payoff_shift_f) * payoff_units_f # linear translation

            tplay_win_payoff = (tplay_win_payoff - t_payoff_dist_chars['mean']) / t_payoff_dist_chars['stdev'] + payoff_shift_f
            tplay_lose_payoff = (tplay_lose_payoff - t_payoff_dist_chars['mean']) / t_payoff_dist_chars['stdev'] + payoff_shift_f
            tfold_lose_payoff = (tfold_lose_payoff - t_payoff_dist_chars['mean']) / t_payoff_dist_chars['stdev'] + payoff_shift_f

            t_choice_options = [Option(name='play', outcomes={'win': {'payoff': tplay_win_payoff, 'prob': tplay_win_prob},
                                                              'lose': {'payoff': tplay_lose_payoff, 'prob': 1 - tplay_win_prob}}),
                                Option(name='fold', outcomes={'lose': {'payoff': tfold_lose_payoff, 'prob': 1 - tfold_win_prob}})]
            try:
                t_post_loss_bool = (player_f.outcomes[game_num][str(int(hand_num)-1)] < 0)
                t_post_win_bool = (player_f.outcomes[game_num][str(int(hand_num) - 1)] > 0)
                t_neutral_bool = (not t_post_loss_bool) and (not t_post_win_bool)
                t_post_loss_xonlyblind_previous_bool = t_post_loss_bool & (
                        player_f.outcomes[game_num][str(int(hand_num)-1)] != -small_blind) & (
                        player_f.outcomes[game_num][str(int(hand_num)-1)] != -big_blind)
                t_post_win_outcome_xonlyblind_previous_bool = t_post_win_bool & (
                            player_f.outcomes[game_num][str(int(hand_num)-1)] != (big_blind + small_blind))
                t_neutral_xonlyblind_bool = (not t_post_loss_xonlyblind_previous_bool) & (not t_post_win_outcome_xonlyblind_previous_bool)
            except KeyError:
                t_post_loss_bool = None
                t_post_win_bool = None
                t_neutral_bool = None
                t_post_loss_xonlyblind_previous_bool = None
                t_post_win_outcome_xonlyblind_previous_bool = None
                t_neutral_xonlyblind_bool = None

            t_choice_situation = ChoiceSituation(sit_options=t_choice_options[:],
                                                 sit_choice="fold" if player_f.actions[game_num][hand_num]['preflop']['final'] == 'f' else "play",
                                                 slansky_strength=player_f.odds[game_num][hand_num]['slansky'],
                                                 stack_rank=player_f.stack_ranks[game_num][hand_num],
                                                 seat=player_f.seat_numbers[game_num][hand_num],
                                                 post_loss=t_post_loss_bool,
                                                 post_win=t_post_win_bool,
                                                 post_neutral=t_neutral_bool,
                                                 post_loss_excl_blind_only=t_post_loss_xonlyblind_previous_bool,
                                                 post_win_excl_blind_only=t_post_win_outcome_xonlyblind_previous_bool,
                                                 post_neutral_or_blind_only=t_neutral_xonlyblind_bool,
                                                 tags={'game': game_num, 'hand': hand_num, 'player': player_f.name, 'id': binascii.b2a_hex(os.urandom(4))}
                                                 )
            choice_situations_f.append(t_choice_situation)
            del t_choice_situation
            del tfold_win_prob, tfold_lose_payoff, tplay_win_payoff, tplay_lose_payoff, t_choice_options
            del t_post_loss_bool, t_post_win_bool, t_neutral_bool, t_post_loss_xonlyblind_previous_bool, t_post_win_outcome_xonlyblind_previous_bool, t_neutral_xonlyblind_bool
        del hand_num, probs
    del game_num, hands
    del big_blind, small_blind, t_payoff_dist_chars, payoff_shift_f

    # t_exp_vals = [cs.options[0].outcomes['win']['prob'] * cs.options[0].outcomes['win']['payoff'] +
    #               cs.options[0].outcomes['lose']['prob'] * cs.options[0].outcomes['lose']['payoff'] +
    #               cs.options[1].outcomes['lose']['prob'] * cs.options[1].outcomes['lose']['payoff']
    #               for cs in choice_situations_f]
    # plt.figure()
    # plt.hist(t_exp_vals, bins=100)
    # plt.title('Choice situation prob/payoff expected values for %s' % player_f.name)

    print('Dropped a total of %d for KeyErrors \n'
          '(likely b/c no observations for combination of slansky/seat/stack for prob and payoff estimates.) \n'
          'Kept a total of %d choice situations' % (num_choice_situations_dropped, len(choice_situations_f)))
    return choice_situations_f


def generate_synthetic_data(add_gamble_type_info_TF=False):
    if not add_gamble_type_info_TF:
        print("Not adding gamble type info for synthetic choice situations to save time, this can be added by setting argument 'add_gamble_type_info_TF' to True")

    def create_synthetic_choice_situations(opt1, opt2, n_cs, act_prop, rank, seat):
        return [ChoiceSituation(sit_options=[opt1, opt2],
                                sit_choice=opt1.name,
                                slansky_strength=rank, stack_rank=1, seat=seat) for i in range(int(n_cs * act_prop))] + \
               [ChoiceSituation(sit_options=[opt1, opt2],
                                sit_choice=opt2.name,
                                slansky_strength=rank, stack_rank=1, seat=seat) for i in range(n_cs - int(n_cs * act_prop))]

    def print_auditing_calcs(t_choice_param_dictionary, t_params):
        for k, v in t_params.items():
            print('Param %s: %s' % (k, v))
        for task in ['cs1', 'cs2', 'cs3', 'cs4']:
            for p in range(1, 11):
                print('TASK %s for p=%3.2f' % (task, p / 10))
                print('Actual choice:   %3.2f' % (
                            t_choice_param_dictionary[int(task.strip('cs'))][p][0]['n_chosen']['play'] / (
                            t_choice_param_dictionary[int(task.strip('cs'))][p][0]['n_chosen']['play'] +
                            t_choice_param_dictionary[int(task.strip('cs'))][p][0]['n_chosen']['fold'])))
                print('Probability RUM:     %3.2f' % calc_RUM_prob(
                    calc_CRRA_utility(t_choice_param_dictionary[int(task.strip('cs'))][p][0]['params']['play'],
                                      t_params['omega']),
                    [calc_CRRA_utility(t_choice_param_dictionary[int(task.strip('cs'))][p][0]['params'][opt],
                                       t_params['omega']) for
                     opt in ['play', 'fold']], t_params['lambda'], t_params['kappa']))
                print('log-likelihood RUM: %3.2f' % - RandomUtilityModel(
                    {int(task.strip('cs')): {p: t_choice_param_dictionary[int(task.strip('cs'))][p]}}).negLL_RUM(
                    [t_params['kappa'], t_params['lambda'], t_params['omega']]))  # / t_total_choices
                print('\n')

    n_obs_dict = {.1: 186, .2: 186, .3: 253, .4: 116, .5: 253, .6: 116, .7: 253, .8: 183, .9: 183, 1: 253}

    act_prop_dict = {
        'cs1': {.1: 0.04, .2: 0.05, .3: 0.1, .4: 0.11, .5: 0.27, .6: 0.44, .7: 0.53, .8: 0.71, .9: 0.87, 1: 0.94},
        'cs2': {.1: 0.06, .2: 0.09, .3: 0.14, .4: 0.21, .5: 0.32, .6: 0.49, .7: 0.61, .8: 0.78, .9: 0.9, 1: 0.96},
        'cs3': {.1: 0.04, .2: 0.06, .3: 0.08, .4: 0.13, .5: 0.21, .6: 0.4, .7: 0.48, .8: 0.63, .9: 0.79, 1: 0.93},
        'cs4': {.1: 0.05, .2: 0.06, .3: 0.09, .4: 0.11, .5: 0.24, .6: 0.43, .7: 0.5, .8: 0.61, .9: 0.78, 1: 0.94}}

    # pred_prop_dict = {
    #     'cs1': {.1: 0.04, .2: 0.05, .3: 0.06, .4: 0.10, .5: 0.18, .6: 0.32, .7: 0.51, .8: 0.70, .9: 0.83, 1: 0.90},
    #     'cs2': {.1: 0.11, .2: 0.15, .3: 0.22, .4: 0.31, .5: 0.43, .6: 0.55, .7: 0.67, .8: 0.77, .9: 0.84, 1: 0.89},
    #     'cs3': {.1: 0.04, .2: 0.05, .3: 0.07, .4: 0.12, .5: 0.21, .6: 0.36, .7: 0.56, .8: 0.73, .9: 0.85, 1: 0.91},
    #     'cs4': {.1: 0.04, .2: 0.05, .3: 0.08, .4: 0.13, .5: 0.22, .6: 0.36, .7: 0.54, .8: 0.71, .9: 0.83, 1: 0.90}}

    master_choice_situations_list = list()
    for p in [(x + 1) / 10 for x in range(10)]:
        cs_opt_spec = {'cs1': {'opt1': Option(name='play', outcomes={'win': {'payoff': 3850, 'prob': p},
                                                                     'lose': {'payoff': 100, 'prob': 1 - p}}),
                               'opt2': Option(name='fold', outcomes={'win': {'payoff': 2000, 'prob': p},
                                                                     'lose': {'payoff': 1600, 'prob': 1 - p}})},
                       'cs2': {'opt1': Option(name='play', outcomes={'win': {'payoff': 4000, 'prob': p},
                                                                     'lose': {'payoff': 500, 'prob': 1 - p}}),
                               'opt2': Option(name='fold', outcomes={'win': {'payoff': 2250, 'prob': p},
                                                                     'lose': {'payoff': 1500, 'prob': 1 - p}})},
                       'cs3': {'opt1': Option(name='play', outcomes={'win': {'payoff': 4000, 'prob': p},
                                                                     'lose': {'payoff': 150, 'prob': 1 - p}}),
                               'opt2': Option(name='fold', outcomes={'win': {'payoff': 2000, 'prob': p},
                                                                     'lose': {'payoff': 1750, 'prob': 1 - p}})},
                       'cs4': {'opt1': Option(name='play', outcomes={'win': {'payoff': 4500, 'prob': p},
                                                                     'lose': {'payoff': 50, 'prob': 1 - p}}),
                               'opt2': Option(name='fold', outcomes={'win': {'payoff': 2500, 'prob': p},
                                                                     'lose': {'payoff': 1000, 'prob': 1 - p}})}
                       }

        choice_situations_dict = {}
        for cs_name, cs in cs_opt_spec.items():
            choice_situations_dict.update({cs_name: create_synthetic_choice_situations(opt1=cs['opt1'], opt2=cs['opt2'],
                                                                                       n_cs=n_obs_dict[p],
                                                                                       act_prop=act_prop_dict[cs_name][
                                                                                           p],
                                                                                       rank=int(cs_name.strip('cs')),
                                                                                       seat=p * 10,
                                                                                       )})  # t_ordered_gamble_info= gamble_type_dict[cs_name][p])

        choice_situations = list()
        for t_cs in choice_situations_dict.values():
            choice_situations = choice_situations + t_cs
        master_choice_situations_list = master_choice_situations_list + choice_situations

    if add_gamble_type_info_TF:
        for cs in master_choice_situations_list:
            cs.CRRA_ordered_gamble_info = cs.get_gamble_info(lambda_list_f=[0.5, 10])
            cs.evaluate_gamble_info()
            cs.find_omega_indifference()

    # ---- synthetic test data -----
    kappa_actual = 0.034  # kappa_RPM = 0.051
    lambda_actual = 0.275  # lambda_RPM = 2.495
    omega_actual = 0.661  # omega_RPM = 0.752

    choice_param_dictionary = reformat_choice_situations_for_model(master_choice_situations_list)

    print_auditing_calcs(choice_param_dictionary, t_params={'kappa': kappa_actual, 'lambda': lambda_actual, 'omega': omega_actual})  #######
    print('Synthetic data dummary: %d total situations, %d play, %d fold' % (
        sum([len(y) for x in choice_param_dictionary.values() for y in x.values()]),
        sum([z['n_chosen']['play'] for x in choice_param_dictionary.values() for y in x.values() for z in y]),
        sum([z['n_chosen']['fold'] for x in choice_param_dictionary.values() for y in x.values() for z in y])))
    print('Actual values of parameters used to generate data\n\tomega = %3.3f, lambda = %3.3f, kappa = %3.3f' % (omega_actual, lambda_actual, kappa_actual))

    return choice_param_dictionary


def reformat_choice_situations_for_model(choice_situations_f, print_obs_summary=True):
    # create dictionary of option params
    choice_param_dictionary_f = {int(rank): {int(seat): [] for seat in set([cs.seat for cs in choice_situations_f])} for
                                 rank in set([cs.slansky_strength for cs in choice_situations_f])}
    for cs in choice_situations_f:
        t_dict = {'params': {}, 'n_chosen': {'play': 0, 'fold': 0}, 'CRRA_gamble_type': None, 'omega_indifference': None, 'id': None}
        for i in range(len(cs.option_names)):
            t_dict['params'].update({cs.option_names[i]: list(cs.options[i].outcomes.values())})
        t_dict['n_chosen'][cs.choice] += 1
        t_dict['CRRA_gamble_type'] = cs.CRRA_ordered_gamble_type
        t_dict['omega_indifference'] = cs.omega_indifference
        t_dict['id'] = cs.tags['id']
        choice_param_dictionary_f[int(cs.slansky_strength)][int(cs.seat)].append(t_dict)


    # if no observations exist for a given rank and seat, drop the dictionary item since we have no information with which to estimate the model
    t_drop_keys = list()
    for rank in choice_param_dictionary_f.keys():
        for seat in choice_param_dictionary_f[rank].keys():
            # ---- old format ------
            # if sum(choice_param_dictionary_f[rank][seat]['n_chosen'].values()) == 0:
            #     t_drop_keys.append((rank, seat))
            #     print('No observations for rank %s seat %s' % (rank, seat))
            #------ old format ------
            if len(choice_param_dictionary_f[rank][seat]) == 0:
                t_drop_keys.append((rank, seat))
                if print_obs_summary:
                    print('No observations for rank %s seat %s' % (rank, seat))
    for pair in t_drop_keys:
        choice_param_dictionary_f[pair[0]].pop(pair[1])

    t_rank_drop_keys = list()
    for rank in choice_param_dictionary_f.keys():
        if len(choice_param_dictionary_f[rank]) == 0:
            t_rank_drop_keys.append(rank)
            print('No observations for rank %s for any seat' % rank)
    for rank in t_rank_drop_keys:
        choice_param_dictionary_f.drop(rank)

    del t_drop_keys, t_rank_drop_keys

    return choice_param_dictionary_f


def calc_CRRA_utility(outcomes, omega):
    def calc_outcome_util(payoff, omega):
        if payoff == 0:
            return 0
        else:
            if omega == 1:
                return np.log(payoff)
            else:
                return (payoff ** (1 - omega)) / (1 - omega)
    return sum([o['prob'] * calc_outcome_util(payoff=o['payoff'], omega=omega) for o in outcomes])


def calc_logit_prob(util_i, util_j, lambda_f):
    if lambda_f is None:
        return 1 / (sum([np.exp(u - util_i) for u in util_j]))
    else:
        return 1 / (sum([np.exp(lambda_f * (u - util_i)) for u in util_j]))


def calc_RUM_prob(util_i, util_j, lambda_f, kappa_f):
    if kappa_f is None:
        return calc_logit_prob(util_i=util_i, util_j=util_j, lambda_f=lambda_f)
    else:
        return (1 - 2 * kappa_f) * calc_logit_prob(util_i, util_j, lambda_f) + kappa_f    ##################


def calc_RPM_prob(util_i, util_j, lambda_f, kappa_f):
    return (1 - 2 * kappa_f) * calc_logit_prob(util_i, util_j, lambda_f) + kappa_f  ##################


def run_multistart(nstart_points, est_param_dictionary, model_object, save_iter=False, save_path=os.getcwd(), save_index_start=0, num_iteration_print=50):
    initial_points = {k: np.random.uniform(low=est_param_dictionary['lb'][k], high=est_param_dictionary['ub'][k], size=nstart_points) for k in est_param_dictionary['lb'].keys()}

    results_list = list()
    for i in range(nstart_points):
        if (i % num_iteration_print) == 0:
            print('Optimizing for starting point %d' % i)
        model_object.fit(init_params=[initial_points[j][i] for j in model_object.param_names_f],
                         est_param_dict=est_param_dictionary)
        results_list.append({'initial_point': {j: initial_points[j][i] for j in model_object.param_names_f},
                             'results': copy.deepcopy(model_object.results), 'prob_model': model_object.LL_form})

        if save_iter is not False:
            if (((i + 1) % save_iter) == 0) and (i > 0):
                try:
                    with open(os.path.join(save_path,
                                           'multistart_results_iter' + str(i + save_index_start + 1 - save_iter) + 't' + str(save_index_start + i)), 'wb') as ff:
                        pickle.dump({'results': results_list, 'model': model_object}, ff)  # pickle.dump(results_list[(i + 1 - save_iter):(i + 1)], ff)
                        results_list = list()
                except FileNotFoundError:
                    try:
                        os.makedirs(save_path)
                        with open(os.path.join(save_path,
                                               'multistart_results_iter' + str(
                                                   i + save_index_start + 1 - save_iter) + 't' + str(
                                                   save_index_start + i)), 'wb') as ff:
                            pickle.dump({'results': results_list, 'model': model_object}, ff)  # pickle.dump(results_list[(i + 1 - save_iter):(i + 1)], ff)
                            results_list = list()
                    except FileExistsError as e:
                        print(e)

    return results_list


def parse_multistart(multistart_results, param_locs_names_f, choice_param_dictionary_f):
    t_param_list_dicts = list()
    t_obs_list_dicts = list()
    for r in multistart_results:
        est_run_id = binascii.b2a_hex(os.urandom(8))
        try:
            t_second_order_cond = pos_def_hess_TF(np.linalg.inv(r['results']['hess_inv'].todense()))
        except:
            t_second_order_cond = None
        try:
            t_sig_results = calc_mle_tstat(r['results'].x, calc_robust_varcov(r['results'].jac, r['results'].hess_inv.todense()))
            t_stderr = t_sig_results['stderr']
            t_tstat = t_sig_results['tstat']
        except:
            t_stderr = None
            t_tstat = None

        try:
            t_lambda = r['results'].x[param_locs_names_f['lambda']]
        except KeyError:
            t_lambda = None
        try:
            t_kappa = r['results'].x[param_locs_names_f['kappa']]
        except:
            t_kappa = None

        # t_param_list_dicts.append({'est_run_id': est_run_id,
        #                            'kappa': r['results'].x[kappa_index],
        #                            'lambda': r['results'].x[lambda_index],
        #                            'omega': r['results'].x[omega_index],
        #                            'message': r['results'].message,
        #                            'pos_def_hess': t_second_order_cond,
        #                            'kappa_initial': r['initial_point']['kappa'],
        #                            'lambda_initial': r['initial_point']['lambda'],
        #                            'omega_initial': r['initial_point']['omega'],
        #                            'kappa_stderr': t_stderr[kappa_index],
        #                            'lambda_stderr': t_stderr[lambda_index],
        #                            'omega_stderr': t_stderr[omega_index],
        #                            'kappa_tstat': t_tstat[kappa_index],
        #                            'lambda_tstat': t_tstat[lambda_index],
        #                            'omega_tstat': t_tstat[omega_index],
        #                            })

        t_dict = {'est_run_id': est_run_id,
         'message': r['results'].message,
         'pos_def_hess': t_second_order_cond}
        t_dict.update({k: r['results'].x[v] for k, v in param_locs_names_f.items()})
        t_dict.update({k +'_initial': r['initial_point'][k] for k in param_locs_names_f.keys()})
        t_dict.update({k + '_stderr': t_stderr[v] for k, v in param_locs_names_f.items()})
        t_dict.update({k + '_tstat': t_tstat[v] for k, v in param_locs_names_f.items()})

        t_param_list_dicts.append(t_dict)
        del t_dict

        for rank in choice_param_dictionary_f.keys():
            for seat in choice_param_dictionary_f[rank].keys():
                for item in choice_param_dictionary_f[rank][seat]:
                    t_util_play = calc_CRRA_utility(item['params']['play'], r['results'].x[param_locs_names_f['omega']])
                    t_util_fold = calc_CRRA_utility(item['params']['fold'], r['results'].x[param_locs_names_f['omega']])
                    t_pred_share = calc_RUM_prob(t_util_play, [t_util_play, t_util_fold],
                                                 t_lambda, t_kappa)
                    t_obs_list_dicts.append(
                        {'est_run_id': est_run_id,
                         'rank': rank,
                         'seat': seat,
                         'actual_share': item['n_chosen']['play'],   # / (item['n_chosen']['play'] + item['n_chosen']['fold']),
                         'pred_share': t_pred_share,
                         'util_play': t_util_play,
                         'util_fold': t_util_fold,
                         'conv_flag': r['results'].status
                         }
                    )

        del t_lambda, t_kappa

    return t_param_list_dicts, t_obs_list_dicts


def pos_def_hess_TF(hess):
    return all(np.sign(np.linalg.eig(hess)[0]) > 0)


def calc_robust_varcov(grad, hess):
    return np.matmul(np.matmul(np.matmul(hess, np.reshape(grad, (len(grad), 1))), np.reshape(grad, (1, len(grad)))), hess)


def calc_mle_tstat(param, varcov):
    if (np.shape(varcov)[0] == len(param)) ^ (np.shape(varcov)[1] == len(param)):
        t_stderr = varcov
    elif (np.shape(varcov)[0] == len(param)) and (np.shape(varcov)[1] == len(param)):
        t_stderr = [np.sqrt(x) for x in np.diag(varcov)]
    else:
        print('Error calculating calc_mle_tstat. varcov not of acceptable dimension')
    return {'param': param, 'stderr': t_stderr, 'tstat': [p/s for p, s in zip(param, t_stderr)]}


def check_likelihood(model_f, lb_f, ub_f, prob_type_f="RUM"):
    # fixomega = 1.01    #model_f.results.x[0]
    # fixlambda = model_f.results.x[1]
    print('Likelihood for probability calculation type %s' % prob_type_f)
    div_factor = 100
    num_points = 20
    # lambda_range = [l / div_factor for l in range(int(lb_f['lambda']*div_factor), int(ub_f['lambda']*div_factor), int(div_factor/num_points))] # [l / div_factor for l in range(0, 200, 10)]
    lambda_range = [o / div_factor for o in range(int(lb_f['lambda'] * div_factor), int(ub_f['lambda'] * div_factor),
                                                  int((int(ub_f['lambda']*div_factor) - int(lb_f['lambda']*div_factor)) / num_points))]
    omega_range = [o / div_factor for o in range(int(lb_f['omega']*div_factor), int(ub_f['omega']*div_factor), int(int(ub_f['omega']*div_factor - int(lb_f['omega']*div_factor)) / num_points))]
    kappa_range = [o / div_factor for o in range(int(lb_f['kappa']*div_factor), int(ub_f['kappa']*div_factor), int(div_factor/num_points))]

    if 'kappa' in model_f.param_names_f:
        # Examine kappa
        test_LL_fixomega_lambda = list()
        for fo in omega_range:
            if (prob_type_f == 'RUM'):
                test_LL_fixomega_lambda.append([model_f.negLL_RUM([fo, 3, k]) for k in kappa_range])
            elif (prob_type_f == 'RPM'):
                test_LL_fixomega_lambda.append([model_f.negLL_RPM([fo, 3, k]) for k in kappa_range])

        plt.figure()
        for s in test_LL_fixomega_lambda:
            plt.plot(kappa_range, s)
        plt.title('Negative LL holding omega fixed at various vals (lambda const)')  # estimate %3.2f' % fixomega)
        plt.xlabel('Kappa')
        plt.legend(['om=' + str(round(o, 2)) for o in omega_range])

    if 'kappa' not in model_f.param_names_f:
        # Examine lambda
        test_LL_fixomega = list()
        for fo in omega_range:
            if (prob_type_f == 'RUM'):
                test_LL_fixomega.append([model_f.negLL_RUM([fo, l]) for l in lambda_range])
            elif (prob_type_f == 'RPM'):
                test_LL_fixomega.append([model_f.negLL_RPM([fo, l]) for l in lambda_range])

        plt.figure()
        for s in test_LL_fixomega:
            plt.plot(lambda_range, s)
        plt.title('Negative LL holding omega fixed at various vals')  # estimate %3.2f' % fixomega)
        plt.xlabel('Lambda')
        plt.legend(['om=' + str(round(o, 2)) for o in omega_range])
        plt.show()

        # Examine omega
        test_LL_fixlambda = list()
        for fl in lambda_range:
            if (prob_type_f == 'RUM'):
                test_LL_fixlambda.append([model_f.negLL_RUM([o, fl]) for o in omega_range])
            elif (prob_type_f == 'RPM'):
                test_LL_fixlambda.append([model_f.negLL_RPM([o, fl]) for o in omega_range])

        plt.figure()
        for s in test_LL_fixlambda:
            plt.plot(omega_range, s)
        plt.title('Negative LL holding lambda fixed at various vals')  # estimate %3.2f' % fixlambda)
        plt.xlabel('omega')
        plt.legend(['lam=' + str(round(l, 2)) for l in lambda_range])
        plt.show()

        #############
        # 2D
        # plt.figure()
        # xx, yy = np.mgrid[omega_range, lambda_range]
        #
        # # Extract x and y
        # xx, yy = np.mgrid[min(omega_range):max(omega_range):100j, min(lambda_range):max(lambda_range):100j]
        # positions = np.vstack([xx.ravel(), yy.ravel()])
        # values = np.vstack([omega_range, lambda_range])
        # f = [model_f.negLL_RUM(positions[:, i]) for i in range(positions.shape[1])]
        # F = np.reshape(f(positions).T, xx.shape)
        #
        # plt.contour(xx, yy, test_LL_fixlambda, cmap='coolwarm')
        ####################
    return None
