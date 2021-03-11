import pickle
import os
import json
import copy
import numpy as np
from scipy.optimize import minimize, Bounds, newton_krylov

from assumption_calc_functions import calc_prob_winning_slansky_rank
from assumption_calc_functions import create_game_hand_index

# ----- File I/O params -----
fp_output = 'output'
fn_prob_payoff_dict = 'prob_payoff_dicts.json'

# ---- Params -----
select_player = 'Bill'

# ----- LOAD DATA -----
# game data
with open("python_hand_data.pickle", 'rb') as f:
    data = pickle.load(f)
players = data['players']
games = data['games']

# probability and payoffs by seat and slansky rank
try:
    with open(os.path.join(fp_output, fn_prob_payoff_dict), 'r') as f:
        t_dict = json.load(f)
        prob_dict = t_dict['prob_dict']
        payoff_dict = t_dict['payoff_dict']
        del t_dict
except FileNotFoundError:
    print('No probability and payoff dictionaries saved down in output folder')
    prob_dict, payoff_dict = calc_prob_winning_slansky_rank(games, slansky_groups_f=None, seat_groups_f=None, stack_groups_f=None)

# ---- Calculations -----
select_player_index = [i for i in range(0, len(players)) if players[i].name == select_player][0]
game_hand_player_index = create_game_hand_index(players[select_player_index])


class Option:
    def __init__(self, name, outcomes):
        self.name = name
        self.outcomes = outcomes


class ChoiceSituation:
    def __init__(self, sit_options, sit_choice=None, slansky_strength=None, stack_rank=None, seat=None, post_loss=None, CRRA_ordered_gamble_type=None):
        self.options = sit_options
        self.option_names = [x.name for x in sit_options]
        self.choice = sit_choice
        self.slansky_strength = slansky_strength
        self.stack_rank = stack_rank
        self.seat = seat
        self.post_loss = post_loss
        self.CRRA_ordered_gamble_type = CRRA_ordered_gamble_type

    def print(self):
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))


class RandomUtilityModel:
    def __init__(self, data_f):
        self.data_f = data_f
        self.kappa_f = None
        self.lambda_f = None
        self.omega_f = None
        self.param_names_f = ['kappa', 'lambda', 'omega']
        self.init_params = None
        self.results = None

        # ---- Manual LL and score ----
        # replaces default loglikelihood for each observation
    def negLL_RUM(self, params):
        def calc_LLi(X, Y, I, util_X, util_Y, kappa, lam):
            return (X + I / 2) * np.log(((1 - 2 * kappa) * np.exp(lam * util_X))
                                        / (np.exp(lam * util_X) + np.exp(lam * util_Y)) + kappa) \
                   + (Y + I / 2) * np.log(((1 - 2 * kappa) * np.exp(lam * util_Y))
                                          / (np.exp(lam * util_X) + np.exp(lam * util_Y)) + kappa)

        self.kappa_f = params[self.param_names_f.index('kappa')]
        self.lambda_f = params[self.param_names_f.index('lambda')]
        self.omega_f = params[self.param_names_f.index('omega')]

        LLi = list()
        for rank in self.data_f.keys():
            for seat in self.data_f[rank].keys():
                LLi.append(calc_LLi(X=self.data_f[rank][seat]['n_chosen']['play'],
                                    Y=self.data_f[rank][seat]['n_chosen']['fold'],
                                    I=0,
                                    util_X=calc_CRRA_utility(outcomes=self.data_f[rank][seat]['params']['play'], omega=self.omega_f),
                                    util_Y=calc_CRRA_utility(outcomes=self.data_f[rank][seat]['params']['fold'], omega=self.omega_f),
                                    kappa=self.kappa_f,
                                    lam=self.lambda_f))

        return -sum(LLi)

    def negLL_RPM(self, params):
        def calc_LLi(X, Y, I, omegaxy, omega, kappa, lam, CRRA_order_type):
            if CRRA_order_type == 'ordered':
                return (X + I / 2) * np.log(((1 - 2 * kappa) * np.exp(lam * omegaxy)) / (
                        np.exp(lam * omegaxy) + np.exp(lam * omega)) + kappa) + \
                       (Y + I / 2) * np.log(((1 - 2 * kappa) * np.exp(lam * omega)) / (
                        np.exp(lam * omegaxy) + np.exp(lam * omega)) + kappa)
            elif CRRA_order_type == 'dominant':
                # if x has higher utility than y for every value of omega
                return (X + I / 2) * np.log(1 - kappa) + (Y + I / 2) * np.log(kappa)
            else:
                print('Error: unknown CRRA gamble type of %s in calculating LL. CHECK.' % CRRA_order_type)
                return None

        self.kappa_f = params[self.param_names_f.index('kappa')]
        self.lambda_f = params[self.param_names_f.index('lambda')]
        self.omega_f = params[self.param_names_f.index('omega')]

        LLi = list()
        for rank in self.data_f.keys():
            for seat in self.data_f[rank].keys():
                t_risky = self.data_f[rank][seat]['CRRA_risky_gamble'] if self.data_f[rank][seat]['CRRA_gamble_type'] == 'ordered' else 'play'
                t_safe = [x for x in self.data_f[rank][seat]['params'].keys() if x != self.data_f[rank][seat]['CRRA_risky_gamble']][0] if self.data_f[rank][seat]['CRRA_gamble_type'] == 'ordered' else 'fold'
                t_omegaxy = self.data_f[rank][seat]['omega_equiv_util'][0] if self.data_f[rank][seat]['CRRA_gamble_type'] == 'ordered' else None
                LLi.append(calc_LLi(X=self.data_f[rank][seat]['n_chosen'][t_risky],
                                    Y=self.data_f[rank][seat]['n_chosen'][t_safe],
                                    I=0,
                                    omegaxy=t_omegaxy,
                                    omega=self.omega_f,
                                    kappa=self.kappa_f,
                                    lam=self.lambda_f,
                                    CRRA_order_type=self.data_f[rank][seat]['CRRA_gamble_type']))

        return -sum(LLi)

    def fit(self, init_params=None, LL_form='RUM', **kwargs):
        if init_params is None:
            self.init_params = [1] * len(self.param_names_f)
        else:
            self.init_params = init_params

        # print('---Arguments passed to solver---')
        # print('Function: %s' % self.negLL)
        # print('Initial point: %s' % self.init_params)
        # for k, v in kwargs.items():
        #     print("%s, %s" % (k, v))

        if LL_form == 'RPM':
            t_LL = self.negLL_RPM
        else:
            t_LL = self.negLL_RUM

        self.results = minimize(t_LL, self.init_params, **kwargs)

    def print(self):
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))


def generate_choice_situations(player_f, game_hand_index_f, prob_dict_f, payoff_dict_f):
    def get_min_observed_payoff(player_ff, game_hand_index_ff):
        # get payoffs to examine min payoff for shifting into postive domain
        obs_avg_payoffs = list()
        for game_num, hands in game_hand_index_ff.items():
            for hand_num in hands:
                try:
                    t_slansky_rank = str(player_ff.odds[game_num][hand_num]['slansky'])
                    t_seat_num = str(player_ff.seat_numbers[game_num][hand_num])
                    obs_avg_payoffs.append(payoff_dict_f[t_slansky_rank][t_seat_num]['win_sum'] / payoff_dict_f[t_slansky_rank][t_seat_num]['win_count'])
                    obs_avg_payoffs.append(payoff_dict_f[t_slansky_rank][t_seat_num]['loss_sum'] / payoff_dict_f[t_slansky_rank][t_seat_num]['loss_count'])
                except KeyError:
                    print('Error for keys game %s and hand %s' % (game_num, hand_num))
        return min(obs_avg_payoffs)

    choice_situations_f = list()
    num_choice_situations_dropped = 0

    big_blind = 100
    small_blind = 50
    payoff_units_f = 1/big_blind
    payoff_shift_f = get_min_observed_payoff(player_f, game_hand_index_f) * -1

    for game_num, hands in game_hand_index_f.items():
        for hand_num in hands:
            try:
                t_slansky_rank = str(player_f.odds[game_num][hand_num]['slansky'])
                t_seat_num = str(player_f.seat_numbers[game_num][hand_num])

                # --- aggregate to rank level
                tplay_win_prob = prob_dict_f[t_slansky_rank][t_seat_num]['win'] / prob_dict_f[t_slansky_rank][t_seat_num]['play_count']
                tplay_win_payoff = payoff_dict_f[t_slansky_rank][t_seat_num]['win_sum'] / payoff_dict_f[t_slansky_rank][t_seat_num]['win_count']
                tplay_lose_payoff = payoff_dict_f[t_slansky_rank][t_seat_num]['loss_sum'] / payoff_dict_f[t_slansky_rank][t_seat_num]['loss_count']

                tfold_win_prob = 0  # cannot win under folding scenario
                if player_f.blinds[game_num][hand_num]['big']:
                    tfold_lose_payoff = (big_blind * -1)
                elif player_f.blinds[game_num][hand_num]['small']:
                    tfold_lose_payoff = (small_blind * -1)
                else:
                    tfold_lose_payoff = 0

                # --- shift/scale payoffs ----
                tplay_win_payoff = (tplay_win_payoff + payoff_shift_f) * payoff_units_f
                tplay_lose_payoff = (tplay_lose_payoff + payoff_shift_f) * payoff_units_f
                tfold_lose_payoff = (tfold_lose_payoff + payoff_shift_f) * payoff_units_f

                t_choice_options = [Option(name='play', outcomes={'win': {'payoff': tplay_win_payoff, 'prob': tplay_win_prob},
                                                                  'lose': {'payoff': tplay_lose_payoff, 'prob': 1 - tplay_win_prob}}),
                                    Option(name='fold', outcomes={'lose': {'payoff': tfold_lose_payoff, 'prob': 1 - tfold_win_prob}})]
                try:
                    t_post_loss_bool = (player_f.outcomes[game_num][str(int(hand_num)-1)] < 0)
                except KeyError:
                    t_post_loss_bool = None

                t_CRRA_ordered_gamble_type = test_for_ordered_gamble_type(gamble1=[v for _, v in t_choice_options[0].outcomes.items()], gamble2=[v for _, v in t_choice_options[1].outcomes.items()])

                t_choice_situation = ChoiceSituation(sit_options=t_choice_options[:],
                                                     sit_choice="fold" if player_f.actions[game_num][hand_num]['preflop'] == 'f' else "play",
                                                     slansky_strength=player_f.odds[game_num][hand_num]['slansky'],
                                                     stack_rank=player_f.stack_ranks[game_num][hand_num],
                                                     seat=player_f.seat_numbers[game_num][hand_num],
                                                     post_loss=t_post_loss_bool,
                                                     CRRA_ordered_gamble_type=t_CRRA_ordered_gamble_type)

                choice_situations_f.append(t_choice_situation)

                del t_choice_situation
            except KeyError:
                num_choice_situations_dropped += 1
    del game_num, hands, hand_num, t_slansky_rank, t_seat_num
    print('Dropped a total of %d for KeyErrors \n(likely b/c no observations for combination of slansky/seat/stack for prob and payoff estimates.) \nKept a total of %d choice situations' % (num_choice_situations_dropped, len(choice_situations_f)))
    return choice_situations_f


def create_synthetic_choice_situations(opt1, opt2, omega_act, lambda_act, kappa_act, n_cs, rank, seat):
    util_opt1 = calc_CRRA_utility(outcomes=opt1.outcomes.values(), omega=omega_act)
    util_opt2 = calc_CRRA_utility(outcomes=opt2.outcomes.values(), omega=omega_act)
    prob_opt1 = calc_RUM_prob(util_opt1, [util_opt1, util_opt2], lambda_act, kappa_act)

    return [ChoiceSituation(sit_options=[opt1, opt2],
                            sit_choice=opt1.name,
                            slansky_strength=rank, stack_rank=1, seat=seat,
                            CRRA_ordered_gamble_type=test_for_ordered_gamble_type(gamble1=[v for _, v in opt1.outcomes.items()],
                                                                                  gamble2=[v for _, v in opt2.outcomes.items()]))
            for i in range(int(n_cs * prob_opt1))] + \
           [ChoiceSituation(sit_options=[opt1, opt2],
                            sit_choice=opt2.name,
                            slansky_strength=rank, stack_rank=1, seat=seat,
                            CRRA_ordered_gamble_type=test_for_ordered_gamble_type(gamble1=[v for _, v in opt1.outcomes.items()],
                                                                                  gamble2=[v for _, v in opt2.outcomes.items()]))
    for i in range(n_cs - int(n_cs * prob_opt1))]


def reformat_choice_situations_for_model(choice_situations):
    # create dictionary of option params
    choice_param_dictionary = {rank: {seat: {'params': dict(), 'n_chosen': {'play': 0, 'fold': 0}, 'CRRA_gamble_type': None, 'CRRA_risky_gamble': None} for seat in set([cs.seat for cs in choice_situations])} for rank in set([cs.slansky_strength for cs in choice_situations])}
    for cs in choice_situations:
        for i in range(len(cs.option_names)):
            choice_param_dictionary[cs.slansky_strength][cs.seat]['params'].update(
                {cs.option_names[i]: list(cs.options[i].outcomes.values())})
        choice_param_dictionary[cs.slansky_strength][cs.seat]['n_chosen'][cs.choice] += 1
        choice_param_dictionary[cs.slansky_strength][cs.seat]['CRRA_gamble_type'] = cs.CRRA_ordered_gamble_type['type']
        if cs.CRRA_ordered_gamble_type['type'] == 'ordered':
            choice_param_dictionary[cs.slansky_strength][cs.seat]['omega_equiv_util'] = cs.CRRA_ordered_gamble_type['cross points']
            # compare utilities as low levels of risk aversion to determine which is risker gamble
            u_play = calc_CRRA_utility(choice_param_dictionary[cs.slansky_strength][cs.seat]['params']['play'], cs.CRRA_ordered_gamble_type['cross points'][0] / 2)
            u_fold = calc_CRRA_utility(choice_param_dictionary[cs.slansky_strength][cs.seat]['params']['fold'], cs.CRRA_ordered_gamble_type['cross points'][0] / 2)
            if u_play > u_fold:
                choice_param_dictionary[cs.slansky_strength][cs.seat]['CRRA_risky_gamble'] = 'play'
            else:
                choice_param_dictionary[cs.slansky_strength][cs.seat]['CRRA_risky_gamble'] = 'fold'

    return choice_param_dictionary


def calc_CRRA_utility(outcomes, omega):
    def calc_outcome_util(prob, payoff, omega):
        if payoff == 0:
            return 0
        else:
            if omega == 1:
                return np.log(payoff)
            else:
                return prob * (np.sign(payoff) * (abs(payoff) ** (1 - omega))) / (1 - omega)
    return sum([calc_outcome_util(prob=o['prob'], payoff=o['payoff'], omega=omega) for o in outcomes])


def calc_CRRA_utility_omegaxy(gamble1, gamble2, init_guess=0.5):
    # formatted function for root finding
    try:
        return float(newton_krylov(lambda x: calc_CRRA_utility(outcomes=gamble1, omega=x) -
                                       calc_CRRA_utility(outcomes=gamble2, omega=x),
                                   xin=init_guess))
    except:
        print('Could not find omega such that Ux = Uy')
        return None


def calc_RUM_prob(util_i, util_j, lambda_f, kappa_f):
    return ((1 - 2 * kappa_f) * (np.exp(lambda_f * util_i))) / sum([np.exp(lambda_f * u) for u in util_j]) + kappa_f


def test_for_ordered_gamble_type(gamble1, gamble2, plot_ordered_gambles_TF=False):
    uplay_greater_ufold = [calc_CRRA_utility(outcomes=gamble1, omega=w / 100)
                           > calc_CRRA_utility(outcomes=gamble2, omega=w / 100) for w in range(0, 100)]
    num_cross_points = sum(
        [uplay_greater_ufold[i + 1] != uplay_greater_ufold[i] for i in range(len(uplay_greater_ufold) - 1)])
    loc_cross_points = [[w / 100 for w in range(0, 100)][i] for i in range(len(uplay_greater_ufold) - 1) if
                        uplay_greater_ufold[i + 1] != uplay_greater_ufold[i]]

    # plot ordered gamble utilities
    if plot_ordered_gambles_TF:
        if (sum(uplay_greater_ufold) != 0) and (sum(uplay_greater_ufold) != len(uplay_greater_ufold)):
            plt.figure()
            plt.plot([w / 100 for w in range(0, 100)],
                     [calc_CRRA_utility(outcomes=gamble1,
                                        omega=w / 100) for w in range(0, 100)])
            plt.plot([w / 100 for w in range(0, 100)],
                     [calc_CRRA_utility(outcomes=gamble2,
                                        omega=w / 100) for w in range(0, 100)])
            plt.title(
                'num. cross points: %d at w=%s' % (num_cross_points, loc_cross_points))
            plt.show()

    if num_cross_points >= 1:
        omegaxy = [calc_CRRA_utility_omegaxy(gamble1, gamble2, init_guess=loc_cross_points[i]) for i in
                   range(len(loc_cross_points))]
        Uxy = [calc_CRRA_utility(outcomes=gamble1, omega=w) for w in omegaxy]
    else:
        omegaxy = None
        Uxy = None

    if num_cross_points == 1:
        return {'type': "ordered", 'cross points': omegaxy, 'utility_cross': Uxy}
    elif num_cross_points == 0:
        return {'type': "dominant", 'cross points': omegaxy, 'utility_cross': Uxy}
    elif num_cross_points > 1:
        return {'type': "multiple intersections", 'cross points': omegaxy, 'utility_cross': Uxy}
    else:
        return "error in test_for_ordered_gamble_type function"


# ====== Run model fitting =======
# ---- actual data ----
# choice_situations = generate_choice_situations(player_f=players[select_player_index], game_hand_index_f=game_hand_player_index, payoff_dict_f=payoff_dict, prob_dict_f=prob_dict)

# choice_param_dictionary = reformat_choice_situations_for_model(choice_situations)

# ---- synthetic test data -----
omega_actual = 1.109
lambda_actual = 1   #9.08
kappa_actual = 0.005    #.005
n_obs = 1000

cs_opt_spec = {'cs1': {'opt1': Option(name='play', outcomes={'win': {'payoff': 3850, 'prob': 0.3}, 'lose': {'payoff': 100, 'prob': 0.7}}),
                       'opt2': Option(name='fold', outcomes={'win': {'payoff': 2000, 'prob': 0.3}, 'lose': {'payoff': 1600, 'prob': 0.7}})},
               'cs2': {'opt1': Option(name='play', outcomes={'win': {'payoff': 4000, 'prob': 0.3}, 'lose': {'payoff': 500, 'prob': 0.7}}),
                       'opt2': Option(name='fold', outcomes={'win': {'payoff': 2250, 'prob': 0.3}, 'lose': {'payoff': 1500, 'prob': 0.7}})}
               }

choice_situations = list()
for cs_name, cs in cs_opt_spec.items():
    choice_situations = choice_situations + create_synthetic_choice_situations(opt1=cs['opt1'], opt2=cs['opt2'],
                                                                               omega_act=omega_actual, lambda_act=lambda_actual, kappa_act=kappa_actual,
                                                                               n_cs=n_obs, rank=1, seat=int(cs_name.strip('cs')))
choice_param_dictionary = reformat_choice_situations_for_model(choice_situations)

# ----- Model fitting

test = RandomUtilityModel(choice_param_dictionary)
test.negLL_RPM([0.1, 2, 0.5])
test.negLL_RUM([0.1, 2, 0.5])
test.fit(init_params=[kappa_actual+.002, lambda_actual + 1, omega_actual + .2], LL_form='RUM',
         method='l-bfgs-b',
         bounds=Bounds(lb=[0, 0, 0], ub=[1, 100, 100]),
         tol=1e-8,
         options={'disp': True, 'maxiter': 1000, 'verbose': 3})
test.print()

# MULTISTART
nstart_points = 1000
initial_points = {'kappa': np.random.uniform(low=0, high=0.1, size=nstart_points),
                  'lambda': np.random.uniform(low=1, high=100, size=nstart_points),
                  'omega': np.random.uniform(low=1, high=2, size=nstart_points)}
[test.negLL([initial_points[test.param_names_f[0]][i],
             initial_points[test.param_names_f[1]][i],
             initial_points[test.param_names_f[2]][i]]) for i in range(len(initial_points['kappa']))]
results_list = list()
for i in range(len(initial_points['kappa'])):
    if (i % 50) == 0:
        print('Optimizing for starting point %d' % i)
    test.fit(init_params=[initial_points[test.param_names_f[0]][i],
                          initial_points[test.param_names_f[1]][i],
                          initial_points[test.param_names_f[2]][i]],
             method='l-bfgs-b',
             bounds=Bounds(lb=[0, 0, 0], ub=[1, 100, 100]),
             tol=1e-8,
             options={'disp': False, 'verbose': 3})
    results_list.append(copy.deepcopy(test.results))
est_dict = {test.param_names_f[0]: [],
            test.param_names_f[1]: [],
            test.param_names_f[2]: []}
for r in range(len(results_list)):
    for p in range(len(test.param_names_f)):
        est_dict[test.param_names_f[p]].append(results_list[r].x[p])

    if results_list[r].x[0] != 1:
        print('%s: %s' % ([initial_points[test.param_names_f[0]][r],
                           initial_points[test.param_names_f[1]][r],
                           initial_points[test.param_names_f[2]][r]], results_list[r].x))

# --------- VISUAL INVESTIGATION ---------
import matplotlib.pyplot as plt
plt.hist([x for x in est_dict[test.param_names_f[2]] if x < 100])
plt.title(test.param_names_f[2])


def test_CRRA_calc(data, x):
    try:
        return calc_CRRA_utility(data, x)
    except ZeroDivisionError:
        return 100


def test_negLL_calc(model_class, x):
    try:
        model_class.negLL([0.008, 4.5, x])
    except ZeroDivisionError:
        return 100


def calc_prob(outcomes_x, outcomes_y, lambda_f, omega_f):
    Ux = test_CRRA_calc(data=outcomes_x, x=omega_f)
    Uy = test_CRRA_calc(data=outcomes_y, x=omega_f)
    return np.exp(lambda_f * Ux) / (np.exp(lambda_f * Ux) + np.exp(lambda_f * Uy))


plt.scatter([x for x in [i/100 for i in range(1, 300)]], [test_CRRA_calc(choice_param_dictionary[5][6]['params']['play'], x) for x in [i/100 for i in range(1, 300)]])

outcomes_x = [{'payoff': 1, 'prob': 0.9}, {'payoff': -60, 'prob': 0.1}]

outcomes_y = [{'payoff': 0, 'prob': 1}]

calc_CRRA_utility(outcomes=outcomes_x, omega=0.4)

plt.scatter([x for x in [i/100 for i in range(0, 100)]], [test_CRRA_calc(data=outcomes_x, x=x) for x in [i/100 for i in range(0, 100)]])
plt.scatter([x for x in [i/100 for i in range(0, 100)]], [test_CRRA_calc(data=outcomes_y, x=x) for x in [i/100 for i in range(0, 100)]])
plt.scatter([x for x in [i/100 for i in range(1, 100)]], [test_CRRA_calc(data=outcomes_x, x=x) - test_CRRA_calc(data=outcomes_y, x=x) for x in [i/100 for i in range(1, 100)]])
plt.scatter([x for x in [i/100 for i in range(1, 100)]], [calc_prob(outcomes_x, outcomes_y, lambda_f=1, omega_f=x) for x in [i/100 for i in range(1, 100)]])

plt.scatter([x for x in [i/100 for i in range(1, 300)]], [test_negLL_calc(test, x) for x in [i/100 for i in range(1, 300)]])


# ---------- ARCHIVE -------
# data frame for sanity checking / working
# df_choices = pd.DataFrame(columns=['slansky', 'seat', 'choice', 'post_loss'])
# for cs in choice_situations:
#     df_choices = df_choices.append(dict(zip(['slansky', 'seat', 'choice', 'post_loss'], [cs.slansky_strength, cs.seat, cs.choice, cs.post_loss])), ignore_index=True)
