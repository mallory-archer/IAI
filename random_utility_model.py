import pickle
import os
import json
import copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
import binascii
import random


from assumption_calc_functions import calc_prob_winning_slansky_rank
from assumption_calc_functions import create_game_hand_index

# ----- File I/O params -----
fp_output = 'output'
fn_prob_payoff_dict = 'prob_payoff_dicts.json'

# ---- Params -----
# --- data selection params
select_player = 'Pluribus'
select_case = 'post_loss_excl_blind_only'  #'post_loss' #'post_loss_excl_blind_only'  # options: post_loss, post_win, post_loss_excl_blind_only, post_win_excl_blind_only, post_neutral, post_neutral_or_blind_only
fraction_of_data_to_use_for_estimation = .8
save_path = os.path.join('output', 'iter_multistart_saves', select_player.lower(), select_case)

# ---- multi start params
num_multistarts = 100000
save_TF = False
save_iter = 10000

# ----- Calcs -----
if not save_TF:
    save_iter = False   # pass the false condition on saving to multifactor function

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
    def __init__(self, sit_options, sit_choice=None, slansky_strength=None, stack_rank=None, seat=None, post_loss=None, post_win=None, post_neutral=None, post_loss_excl_blind_only=None, post_win_excl_blind_only=None, post_neutral_or_blind_only=None, CRRA_ordered_gamble_type=None):
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

    def negLL_RUM(self, params):
        def calc_LLi(X, Y, I, util_X, util_Y, kappa, lam):
            return (X + I / 2) * np.log(calc_RUM_prob(util_X, [util_X, util_Y], lam, kappa)) + \
                   (Y + I / 2) * np.log(calc_RUM_prob(util_Y, [util_X, util_Y], lam, kappa))

        self.kappa_f = params[self.param_names_f.index('kappa')]
        self.lambda_f = params[self.param_names_f.index('lambda')]
        self.omega_f = params[self.param_names_f.index('omega')]

        LLi = list()
        t_total_obs = 0
        for rank in self.data_f.keys():
            for seat in self.data_f[rank].keys():
                t_LLi = calc_LLi(X=self.data_f[rank][seat]['n_chosen']['play'],
                                 Y=self.data_f[rank][seat]['n_chosen']['fold'],
                                 I=0,
                                 util_X=calc_CRRA_utility(outcomes=self.data_f[rank][seat]['params']['play'], omega=self.omega_f),
                                 util_Y=calc_CRRA_utility(outcomes=self.data_f[rank][seat]['params']['fold'], omega=self.omega_f),
                                 kappa=self.kappa_f,
                                 lam=self.lambda_f)
                LLi.append(t_LLi)
                t_total_obs += sum(self.data_f[rank][seat]['n_chosen'].values())
        return -sum(LLi)/t_total_obs

    # def negLL_RPM(self, params):
    #     print('WARNING: negLL_RPM may not be correct. Could not match Apesteguaia 2018 results. Should not be used without further examination')
    #     def calc_LLi(X, Y, I, omegaxy, omega, kappa, lam, CRRA_order_type):
    #         return (X + I / 2) * np.log(calc_RPM_prob(riskier_TF=True, omegaxy, omega, kappa, lam, CRRA_order_type)) +\
    #                (Y + I / 2) * np.log(calc_RPM_prob(riskier_TF=False, omegaxy, omega, kappa, lam, CRRA_order_type))
    #
    #     self.kappa_f = params[self.param_names_f.index('kappa')]
    #     self.lambda_f = params[self.param_names_f.index('lambda')]
    #     self.omega_f = params[self.param_names_f.index('omega')]
    #
    #     LLi = list()
    #     t_total_obs = 0
    #     for rank in self.data_f.keys():
    #         for seat in self.data_f[rank].keys():
    #             t_risky = 'play'    #### THIS MUST BE UPDATED TO PULL FROM EVALUATION OF GAMBLES
    #             t_not_risky = 'fold'    #### THIS MUST BE UPDATED TO PULL FROM EVALUATION OF GAMBLES
    #             t_omegaxy = 0.5     #### THIS MUST BE UPDATED TO PULL FROM EVALUATION OF GAMBLES
    #             t_CRRA_order_type = 'ordered'     #### THIS MUST BE UPDATED TO PULL FROM EVALUATION OF GAMBLES
    #             LLi.append(calc_LLi(X=self.data_f[rank][seat]['n_chosen'][t_risky],
    #                                 Y=self.data_f[rank][seat]['n_chosen'][t_not_risky],
    #                                 I=0,
    #                                 omegaxy=t_omegaxy,
    #                                 omega=self.omega_f,
    #                                 kappa=self.kappa_f,
    #                                 lam=self.lambda_f,
    #                                 CRRA_order_type=t_CRRA_order_type))
    #             t_total_obs += sum(self.data_f[rank][seat]['n_chosen'].values())
    #     return -sum(LLi)/t_total_obs

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

                # Class ChoiceSituaiton accepts additional specification of "ordered" or "dominant" gamble type,
                # currently do not have ordered vs. dominant type working
                t_choice_situation = ChoiceSituation(sit_options=t_choice_options[:],
                                                     sit_choice="fold" if player_f.actions[game_num][hand_num]['preflop'] == 'f' else "play",
                                                     slansky_strength=player_f.odds[game_num][hand_num]['slansky'],
                                                     stack_rank=player_f.stack_ranks[game_num][hand_num],
                                                     seat=player_f.seat_numbers[game_num][hand_num],
                                                     post_loss=t_post_loss_bool,
                                                     post_win=t_post_win_bool,
                                                     post_neutral=t_neutral_bool,
                                                     post_loss_excl_blind_only=t_post_loss_xonlyblind_previous_bool,
                                                     post_win_excl_blind_only=t_post_win_outcome_xonlyblind_previous_bool,
                                                     post_neutral_or_blind_only=t_neutral_xonlyblind_bool
                                                     )

                choice_situations_f.append(t_choice_situation)

                del t_choice_situation
            except KeyError:
                num_choice_situations_dropped += 1
    del game_num, hands, hand_num, t_slansky_rank, t_seat_num
    print('Dropped a total of %d for KeyErrors \n(likely b/c no observations for combination of slansky/seat/stack for prob and payoff estimates.) \nKept a total of %d choice situations' % (num_choice_situations_dropped, len(choice_situations_f)))
    return choice_situations_f


def generate_synthetic_data():
    def create_synthetic_choice_situations(opt1, opt2, n_cs, act_prop, rank, seat, t_ordered_gamble_type):
        return [ChoiceSituation(sit_options=[opt1, opt2],
                                sit_choice=opt1.name,
                                slansky_strength=rank, stack_rank=1, seat=seat,
                                CRRA_ordered_gamble_type=t_ordered_gamble_type) for i in range(int(n_cs * act_prop))] + \
               [ChoiceSituation(sit_options=[opt1, opt2],
                                sit_choice=opt2.name,
                                slansky_strength=rank, stack_rank=1, seat=seat,
                                CRRA_ordered_gamble_type=t_ordered_gamble_type) for i in
                range(n_cs - int(n_cs * act_prop))]

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

    gamble_type_dict = {'cs' + str(t): {
        x / 10: {'type': 'dominant', 'cross points': [None]} if x == 10 else {'type': 'ordered', 'cross points': [.05]}
        for x in range(1, 11)} for t in range(1, 5)}

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
                                                                                       t_ordered_gamble_type=
                                                                                       gamble_type_dict[cs_name][p])})

        choice_situations = list()
        for t_cs in choice_situations_dict.values():
            choice_situations = choice_situations + t_cs
        master_choice_situations_list = master_choice_situations_list + choice_situations

    return reformat_choice_situations_for_model(master_choice_situations_list)


def reformat_choice_situations_for_model(choice_situations_f):
    # create dictionary of option params
    choice_param_dictionary_f = {rank: {seat: {'params': dict(), 'n_chosen': {'play': 0, 'fold': 0}, 'CRRA_gamble_type': None, 'CRRA_risky_gamble': None} for seat in set([cs.seat for cs in choice_situations_f])} for rank in set([cs.slansky_strength for cs in choice_situations_f])}
    for cs in choice_situations_f:
        for i in range(len(cs.option_names)):
            choice_param_dictionary_f[cs.slansky_strength][cs.seat]['params'].update(
                {cs.option_names[i]: list(cs.options[i].outcomes.values())})
        choice_param_dictionary_f[cs.slansky_strength][cs.seat]['n_chosen'][cs.choice] += 1

    # if no observations exist for a given rank and seat, drop the dictionary item since we have no information with which to estimate the model
    t_drop_keys = list()
    for rank in choice_param_dictionary_f.keys():
        for seat in choice_param_dictionary_f[rank].keys():
            if sum(choice_param_dictionary_f[rank][seat]['n_chosen'].values()) == 0:
                t_drop_keys.append((rank, seat))
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

        ##### need to revise function to evaluate ordered gamble or not
        # choice_param_dictionary_f[cs.slansky_strength][cs.seat]['CRRA_gamble_type'] = cs.CRRA_ordered_gamble_type['type']
        # if cs.CRRA_ordered_gamble_type['type'] == 'ordered':
        #     choice_param_dictionary_f[cs.slansky_strength][cs.seat]['omega_equiv_util'] = cs.CRRA_ordered_gamble_type['cross points']
        #     # compare utilities as low levels of risk aversion to determine which is risker gamble
        #     u_play = calc_CRRA_utility(choice_param_dictionary_f[cs.slansky_strength][cs.seat]['params']['play'], cs.CRRA_ordered_gamble_type['cross points'][0] / 2)
        #     u_fold = calc_CRRA_utility(choice_param_dictionary_f[cs.slansky_strength][cs.seat]['params']['fold'], cs.CRRA_ordered_gamble_type['cross points'][0] / 2)
        #     if u_play > u_fold:
        #         choice_param_dictionary_f[cs.slansky_strength][cs.seat]['CRRA_risky_gamble'] = 'play'
        #     else:
        #         choice_param_dictionary_f[cs.slansky_strength][cs.seat]['CRRA_risky_gamble'] = 'fold'

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
    return 1 / (sum([np.exp(lambda_f * (u - util_i)) for u in util_j]))


def calc_RUM_prob(util_i, util_j, lambda_f, kappa_f):
    return (1 - 2 * kappa_f) * calc_logit_prob(util_i, util_j, lambda_f) + kappa_f


def print_auditing_calcs(t_choice_param_dictionary, t_kappa, t_lambda, t_omega):
    for task in ['cs1', 'cs2', 'cs3', 'cs4']:
        for p in range(1, 11):
            print('TASK %s for p=%3.2f' % (task, p / 10))
            print('Actual choice:   %3.2f' % (t_choice_param_dictionary[int(task.strip('cs'))][p]['n_chosen']['play'] / (
                    t_choice_param_dictionary[int(task.strip('cs'))][p]['n_chosen']['play'] +
                    t_choice_param_dictionary[int(task.strip('cs'))][p]['n_chosen']['fold'])))
            print('Probability RUM:     %3.2f' % calc_RUM_prob(
                calc_CRRA_utility(t_choice_param_dictionary[int(task.strip('cs'))][p]['params']['play'], t_omega),
                [calc_CRRA_utility(t_choice_param_dictionary[int(task.strip('cs'))][p]['params'][opt], t_omega) for
                 opt in ['play', 'fold']], t_lambda, t_kappa))
            print('log-likelihood RUM: %3.2f' % - RandomUtilityModel(
                {int(task.strip('cs')): {p: t_choice_param_dictionary[int(task.strip('cs'))][p]}}).negLL_RUM(
                [t_kappa, t_lambda, t_omega]))  # / t_total_choices
            print('\n')


def run_multistart(nstart_points, t_lb, t_ub, model_object, save_iter=False, save_path=os.getcwd()):
    initial_points = {'kappa': np.random.uniform(low=t_lb['kappa'], high=t_ub['kappa'], size=nstart_points),
                      'lambda': np.random.uniform(low=t_lb['lambda'], high=t_ub['lambda'], size=nstart_points),
                      'omega': np.random.uniform(low=t_lb['omega'], high=t_ub['omega'], size=nstart_points)}
    [model_object.negLL_RUM([initial_points[model_object.param_names_f[0]][i], initial_points[model_object.param_names_f[1]][i],
                     initial_points[model_object.param_names_f[2]][i]]) for i in range(len(initial_points['kappa']))]

    results_list = list()
    for i in range(len(initial_points['kappa'])):
        if (i % 50) == 0:
            print('Optimizing for starting point %d' % i)
        model_object.fit(init_params=[initial_points[model_object.param_names_f[0]][i],
                                      initial_points[model_object.param_names_f[1]][i],
                                      initial_points[model_object.param_names_f[2]][i]],
                         LL_form='RUM',
                         method='l-bfgs-b',
                         bounds=Bounds(lb=[t_lb[model_object.param_names_f[i]] for i in range(len(model_object.param_names_f))], ub=[t_ub[model_object.param_names_f[i]] for i in range(len(model_object.param_names_f))]),
                         options={'disp': False, 'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-5})   # tol=1e-8,
        results_list.append({'initial_point': {model_object.param_names_f[0]: initial_points[model_object.param_names_f[0]][i],
                                         model_object.param_names_f[1]: initial_points[model_object.param_names_f[1]][i],
                                         model_object.param_names_f[2]: initial_points[model_object.param_names_f[2]][i]},
                             'results': copy.deepcopy(model_object.results)})

        if save_iter is not False:
            if (((i + 1) % save_iter) == 0) and (i > 0):
                with open(os.path.join(save_path,
                                       'multistart_results_iter' + str(i +200000 + 1 - save_iter) + 't' + str(200000 + i)), 'wb') as ff:
                    pickle.dump(results_list, ff)  # pickle.dump(results_list[(i + 1 - save_iter):(i + 1)], ff)
                    results_list = list()


    # --- alternative storage format, currently unused
    # est_dict = {model_object.param_names_f[0]: [],
    #             model_object.param_names_f[1]: [],
    #             model_object.param_names_f[2]: []}
    # for r in range(len(results_list)):
    #     for p in range(len(model_object.param_names_f)):
    #         est_dict[model_object.param_names_f[p]].append(results_list[r].x[p])
    #
    #     if results_list[r].x[0] != 1:
    #         print('%s: %s' % ([initial_points[model_object.param_names_f[0]][r],
    #                            initial_points[model_object.param_names_f[1]][r],
    #                            initial_points[model_object.param_names_f[2]][r]], results_list[r].x))

    return results_list


def parse_multistart(multistart_results, kappa_index, lambda_index, omega_index):
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

        t_param_list_dicts.append({'est_run_id': est_run_id,
                                       'kappa': r['results'].x[kappa_index],
                                       'lambda': r['results'].x[lambda_index],
                                   'omega': r['results'].x[omega_index],
                                   'message': r['results'].message,
                                   'pos_def_hess': t_second_order_cond,
                                   'kappa_initial': r['initial_point']['kappa'],
                                   'lambda_initial': r['initial_point']['lambda'],
                                   'omega_initial': r['initial_point']['omega'],
                                   'kappa_stderr': t_stderr[kappa_index],
                                   'lambda_stderr': t_stderr[lambda_index],
                                   'omega_stderr': t_stderr[omega_index],
                                   'kappa_tstat': t_tstat[kappa_index],
                                   'lambda_tstat': t_tstat[lambda_index],
                                   'omega_tstat': t_tstat[omega_index],
                                   })
        for rank in choice_param_dictionary.keys():
            for seat in choice_param_dictionary[rank].keys():
                t_util_play = calc_CRRA_utility(choice_param_dictionary[rank][seat]['params']['play'],
                                                r['results'].x[omega_index])
                t_util_fold = calc_CRRA_utility(choice_param_dictionary[rank][seat]['params']['fold'],
                                                r['results'].x[omega_index])
                t_obs_list_dicts.append(
                    {'est_run_id': est_run_id,
                     'rank': rank,
                     'seat': seat,
                     'actual_share': choice_param_dictionary[rank][seat]['n_chosen']['play'] / (
                                 choice_param_dictionary[rank][seat]['n_chosen']['play'] +
                                 choice_param_dictionary[rank][seat]['n_chosen']['fold']),
                     'pred_share': calc_RUM_prob(t_util_play, [t_util_play, t_util_fold],
                                                 r['results'].x[lambda_index],
                                                 r['results'].x[kappa_index]),
                     'util_play': t_util_play,
                     'util_fold': t_util_fold,
                     'conv_flag': r['results'].status
                     }
                )

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


# ====== Import data ======
# ---- actual data ----
choice_situations = generate_choice_situations(player_f=players[select_player_index], game_hand_index_f=game_hand_player_index, payoff_dict_f=payoff_dict, prob_dict_f=prob_dict)

#######
# ---- debugging, can delete ------
t_counts = {'win': 0, 'loss': 0, 'post_neutral': 0, 'loss_no_blind': 0, 'win_no_blind': 0, 'post_neutral_or_blind_only': 0}
for cs in choice_situations:
    # if cs.post_loss or cs.post_win or cs.post_loss_excl_blind_only or cs.post_win_excl_blind_only:
    #     print(k, v)
    #     break
    if cs.post_loss:
        t_counts['loss'] += 1
    if cs.post_win:
        t_counts['win'] += 1
    if cs.post_neutral:
        t_counts['post_neutral'] += 1
    if cs.post_loss_excl_blind_only:
        t_counts['loss_no_blind'] += 1
    if cs.post_win_excl_blind_only:
        t_counts['win_no_blind'] += 1
    if cs.post_neutral_or_blind_only:
        t_counts['post_neutral_or_blind_only'] += 1
    # print('%s %s' % (k, v))
for k, v in t_counts.items():
    print('%s: %s' % (k, v))
#######

# ---- preprocess: partition data and sub-sample -----
if select_case != 'all':
    t_candidates = [cs for cs in choice_situations if cs.__getattribute__(select_case)]
else:
    t_candidates = choice_situations
choice_param_dictionary = reformat_choice_situations_for_model(random.sample(t_candidates, round(fraction_of_data_to_use_for_estimation * len(t_candidates))))
del t_candidates

# ---- synthetic test data -----
# kappa_actual = 0.034    # kappa_RPM = 0.051
# lambda_actual = 0.275   # lambda_RPM = 2.495
# omega_actual = 0.661    # omega_RPM = 0.752
#
# choice_param_dictionary = generate_synthetic_data()
# print_auditing_calcs(choice_param_dictionary, t_kappa=0.034, t_lambda=0.275, t_omega=0.661)

# ====== Run model fitting =======
# --- create model object
model = RandomUtilityModel(choice_param_dictionary)

#######
# ---- debugging / audititing, can delete
t_sum = list()
for rank in model.data_f.keys():
    for seat in model.data_f[rank].keys():
        t_sum.append(sum(model.data_f[rank][seat]['n_chosen'].values()))
print('Total observations in choice_param_dictionary: %d' % sum(t_sum))
del t_sum
######

# --- fit one model
model.negLL_RUM([.034, 0.275, 0.661])
model.negLL_RUM([.1, 2, 0.5])
lb = {'kappa': 0.000, 'lambda': 0.0000, 'omega': 0.000}
ub = {'kappa': .5, 'lambda': 3, 'omega': 2}
model.fit(init_params=[0.05, .5, .5], LL_form='RUM',
         method='l-bfgs-b',
         bounds=Bounds(lb=[lb[model.param_names_f[i]] for i in range(len(model.param_names_f))], ub=[ub[model.param_names_f[i]] for i in range(len(model.param_names_f))]),
         tol=1e-12,
         options={'disp': True, 'maxiter': 500})    # bounds=Bounds(lb=[0.0001, 0.0001, 0], ub=[10, 10, 15]),
model.print()

# --- run multistart
# run and save
select_results = run_multistart(nstart_points=num_multistarts, t_lb=lb, t_ub=ub, model_object=model, save_iter=save_iter, save_path=save_path)

# load from saved
select_results = list()
for fn in [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f)) if f[0] != '.']:  #os.listdir(save_path):
    with open(os.path.join(save_path, fn), 'rb') as f:
        select_results = select_results + pickle.load(f)

##########
# ---- debugging / auditing, can delete
# i = 0
# eps = 1e-4
# for r in select_results:
#     if (r['results'].x[0] > (lb['kappa'] + eps)) & (r['results'].x[0] < (ub['kappa'] - eps)) & \
#             (r['results'].x[1] > (lb['lambda'] + eps)) & (r['results'].x[1] < (ub['lambda'] - eps)) & \
#             (r['results'].x[2] > (lb['omega'] + eps)) & (r['results'].x[2] < (ub['omega'] - eps)) & \
#             all([x > 1.96 for x in calc_mle_tstat(r['results'].x, calc_robust_varcov(r['results'].jac, r['results'].hess_inv.todense()))['tstat']]):    # & \
#             # (r['results'].message == b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'):
#         print('%d %s' % (i, r['results'].x))
#     i += 1
###########

list_dict_params, list_dict_obs = parse_multistart(select_results, kappa_index=model.param_names_f.index('kappa'), lambda_index=model.param_names_f.index('lambda'), omega_index=model.param_names_f.index('omega'))

# ====== SAVE TO CSV FOR TABLEAU EXAMINATION =======
if save_TF:
    pd.DataFrame(list_dict_params).set_index('est_run_id').to_csv(os.path.join('output', select_player.lower() + '_multistart_params_' + select_case + '.csv'))
    pd.DataFrame(list_dict_obs).set_index(['est_run_id', 'rank', 'seat']).to_csv(os.path.join('output', select_player.lower() + '_multistart_obs_' + select_case + '.csv'))






# ====== ANALYSIS RESULTS =====
from assumption_calc_functions import two_sample_test_ind_means

eps_prop_bound = 0.2
stat_sig_thresh = 1.96
filters = {'kappa_lb': 1e-8, 'omega_lb': 1e-8, 'lambda_lb': 1e-8, 'kappa_ub': 0.4995, 'omega_at_1': eps_prop_bound,
           'kappa_at_initial': eps_prop_bound, 'omega_at_initial': eps_prop_bound, 'lambda_at_initial': eps_prop_bound,
           'omega_stat_sig': stat_sig_thresh, 'lambda_stat_sig': stat_sig_thresh}   # 'kappa_stat_sig': stat_sig_thresh,
test_param_name = 'kappa'


class Scenario:
    def __init__(self, player, case, filters=None):
        self.player = player
        self.case = case
        self.mean = {'filtered_data': {'kappa': None, 'omega': None, 'lambdap': None}, 'data': {'kappa': None, 'omega': None, 'lambdap': None}}
        self.sample_std = {'filtered_data': {'kappa': None, 'omega': None, 'lambdap': None}, 'data': {'kappa': None, 'omega': None, 'lambdap': None}}
        self.n_obs = {'filtered_data': {'kappa': None, 'omega': None, 'lambdap': None}, 'data': {'kappa': None, 'omega': None, 'lambdap': None}}
        self.data = None
        self.filters = filters
        self.filtered_data = None
        
    def import_data(self):
        self.data = pd.read_csv(os.path.join('output', self.player.lower() + '_multistart_params_' + self.case + '.csv')).set_index('est_run_id')
        self.data.rename(columns={'lambda': 'lambdap'}, inplace=True)
    
    def filter_data(self):
        if self.filters is None:
            print('Warning: no filter conditions provided in attribute. Filtered data is identical to original data.')
            self.filtered_data = self.data
        else:
            filter_master = pd.Series(data=[True] * len(self.data.index), index=self.data.index, dtype=bool)
            print('Total valid start obs: %d' % sum(filter_master))
            if 'kappa_lb' in list(self.filters.keys()):
                # filter1 = (self.data.kappa > 1e-8)
                filter_master = filter_master & (self.data.kappa > self.filters['kappa_lb'])
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'omega_lb' in list(self.filters.keys()):
                # filter2 = (self.data.omega > 1e-8)
                filter_master = filter_master & (self.data.omega > self.filters['omega_lb'])
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'lambda_lb' in list(self.filters.keys()):
                # filter3 = (self.data.lambdap > 1e-8)
                filter_master = filter_master & (self.data.lambdap > self.filters['lambda_lb'])
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'kappa_ub' in list(self.filters.keys()):
                # filter4 = (self.data.kappa < 0.4995)
                filter_master = filter_master & (self.data.kappa < self.filters['kappa_ub'])
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'omega_at_1' in list(self.filters.keys()):
                # filter5 = ((self.data.omega < (1 / (1 - eps_prop_bound_f))) & (self.data.omega > (1 * (1 - eps_prop_bound_f))))
                filter_master = filter_master & ~((self.data.omega < (1 / (1 - self.filters['omega_at_1']))) & (
                            self.data.omega > (1 * (1 - self.filters['omega_at_1']))))
                # print('omega at 1 Total valid start obs: %d' % sum(filter_master))
            if 'kappa_at_initial' in list(self.filters.keys()):
                # filter6 = ~((self.data.kappa < (self.data.kappa_initial / (1 - eps_prop_bound_f))) & (
                #                 self.data.kappa > (self.data.kappa_initial * (1 - eps_prop_bound_f))))
                filter_master = filter_master & ~(
                            (self.data.kappa < (self.data.kappa_initial / (1 - self.filters['kappa_at_initial']))) & (
                                self.data.kappa > (self.data.kappa_initial * (1 - self.filters['kappa_at_initial']))))
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'omega_at_initial' in list(self.filters.keys()):
                # filter7 = ~((self.data.omega < (self.data.omega_initial / (1 - eps_prop_bound_f))) & (
                #                 self.data.omega > (self.data.omega_initial * (1 - eps_prop_bound_f))))
                filter_master = filter_master & ~(
                            (self.data.omega < (self.data.omega_initial / (1 - self.filters['omega_at_initial']))) & (
                                self.data.omega > (self.data.omega_initial * (1 - self.filters['omega_at_initial']))))
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'lambda_at_initial' in list(self.filters.keys()):
                # filter8 = ~((self.data.lambdap < (self.data.lambda_initial / (1 - eps_prop_bound_f))) & (
                #                 self.data.lambdap > (self.data.lambda_initial * (1 - eps_prop_bound_f))))
                filter_master = filter_master & ~(
                            (self.data.lambdap < (self.data.lambda_initial / (1 - self.filters['lambda_at_initial']))) & (
                                self.data.lambdap > (self.data.lambda_initial * (1 - self.filters['lambda_at_initial']))))
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'kappa_stat_sig' in list(self.filters.keys()):
                #  filter9 = self.data.kappa_tstat > 0
                filter_master = filter_master & (self.data.kappa_tstat > self.filters['kappa_stat_sig'])
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'omega_stat_sig' in list(self.filters.keys()):
                #  filter10 = self.data.omega_tstat > 1.96
                filter_master = filter_master & (self.data.omega_tstat > self.filters['omega_stat_sig'])
                # print('Total valid start obs: %d' % sum(filter_master))
            if 'lambda_stat_sig' in list(self.filters.keys()):
                #  filter11 = self.data.lambda_tstat > 1.96
                filter_master = filter_master & (self.data.lambda_tstat > self.filters['lambda_stat_sig'])
                # print('Total valid start obs: %d' % sum(filter_master))
    
            self.filtered_data = self.data[filter_master]

    def calc_data_set_sum_stats(self, data_set='data'):
        for dset in ['data', 'filtered_data']:
            for par in ['kappa', 'omega', 'lambdap']:
                self.mean[dset][par] = self.__getattribute__(dset)[par].mean()
                self.sample_std[dset][par] = self.__getattribute__(dset)[par].std()
                self.n_obs[dset][par] = self.__getattribute__(dset)[par].shape[0]


def run_test(test_info_f, test_param_name_f, filtered_type_f, scenarios_f):
    t_scen1_f = scenarios_f[test_info_f['scenario1']['player']][test_info_f['scenario1']['case']]
    t_scen2_f = scenarios_f[test_info_f['scenario2']['player']][test_info_f['scenario2']['case']]

    t_tstat_f, t_pval_f = two_sample_test_ind_means(mu1=t_scen1_f.mean[filtered_type_f][test_param_name_f],
                                                    mu2=t_scen2_f.mean[filtered_type_f][test_param_name_f],
                                                    s1=t_scen1_f.sample_std[filtered_type_f][test_param_name_f],
                                                    s2=t_scen2_f.sample_std[filtered_type_f][test_param_name_f],
                                                    n1=t_scen1_f.n_obs[filtered_type_f][test_param_name_f],
                                                    n2=t_scen2_f.n_obs[filtered_type_f][test_param_name_f],
                                                    n_sides=2)

    print('\nFor param %s:\n(player %s, case %s, mean = %3.4f, stdev = %3.4f, n_obs=%d) vs\n(player %s, case %s, mean = %3.4f, stdev = %3.4f, n_obs=%d):\ntwo-sided diff of means pval = %3.4f' % (
        test_param_name_f,
        test_info_f['scenario1']['player'],
        test_info_f['scenario1']['case'],
        t_scen1_f.mean[filtered_type_f][test_param_name_f],
        t_scen1_f.sample_std[filtered_type_f][test_param_name_f],
        t_scen1_f.n_obs[filtered_type_f][test_param_name_f],
        test_info_f['scenario2']['player'],
        test_info_f['scenario2']['case'],
        t_scen2_f.mean[filtered_type_f][test_param_name_f],
        t_scen2_f.sample_std[filtered_type_f][test_param_name_f],
        t_scen2_f.n_obs[filtered_type_f][test_param_name_f],
        t_pval_f))

    return t_tstat_f, t_pval_f


tests = [{'scenario1': {'player': 'bill', 'case': 'post_neutral_or_blind_only'}, 'scenario2': {'player': 'bill', 'case': 'post_loss_excl_blind_only'}},
         {'scenario1': {'player': 'pluribus', 'case': 'post_neutral_or_blind_only'}, 'scenario2': {'player': 'pluribus', 'case': 'post_loss_excl_blind_only'}}]
unique_sets = list()
for t in tests:
    for s in t.values():
        unique_sets.append((s['player'], s['case']))

scenarios = dict()
for us in set(unique_sets):
    t_scen = Scenario(us[0], us[1], filters)
    t_scen.import_data()
    t_scen.filter_data()
    t_scen.calc_data_set_sum_stats()
    try:
        scenarios[us[0]][us[1]] = t_scen
    except KeyError:
        scenarios.update({us[0]: {us[1]: t_scen}})
    del t_scen

for t in tests:
    tstat, pval = run_test(t, test_param_name, 'filtered_data', scenarios)



#################

# ---------- ARCHIVE -------
# data frame for sanity checking / working
# df_choices = pd.DataFrame(columns=['slansky', 'seat', 'choice', 'post_loss'])
# for cs in choice_situations:
#     df_choices = df_choices.append(dict(zip(['slansky', 'seat', 'choice', 'post_loss'], [cs.slansky_strength, cs.seat, cs.choice, cs.post_loss])), ignore_index=True)


# def calc_CRRA_vutil(outcomes, omega):
#     def calc_outcome_util(payoff, omega):
#         if payoff == 0:
#             return 0
#         else:
#             if omega == 1:
#                 # derivative of CRRA utility undefined at omega = 1
#                 return None
#             else:
#                 return ((payoff ** (1 - omega)) * (1 - ((1 - omega) * np.log(payoff)))) / ((1 - omega) ** 2)
#     try:
#         return sum([o['prob'] * calc_outcome_util(payoff=o['payoff'], omega=omega) for o in outcomes])
#     except TypeError:
#         return None


# calc_CRRA_utility_omegaxy([{'payoff': 1, 'prob': 0.9}, {'payoff': 60, 'prob': 0.1}], [{'payoff': 5, 'prob': 1}])  #####
# def calc_CRRA_utility_omegaxy(gamble1, gamble2, init_guess=0.5):
#     # formatted function for root finding
#     try:
#         # return float(newton_krylov(lambda x: calc_CRRA_vutil(outcomes=gamble1, omega=x) - calc_CRRA_vutil(outcomes=gamble2, omega=x),
#         #                            xin=init_guess))
#         return float(newton_krylov(lambda x: calc_CRRA_utility(outcomes=gamble1, omega=x) -
#                                        calc_CRRA_utility(outcomes=gamble2, omega=x),
#                                    xin=init_guess))
#     except:
#         print('Could not find omega such that Ux = Uy')
#         return None


# test_for_ordered_gamble_type(list(cs_opt_spec[task]['opt1'].outcomes.values()),
#                              list(cs_opt_spec[task]['opt2'].outcomes.values()))['type'] #####
# gamble1 = list(cs_opt_spec[task]['opt1'].outcomes.values())
# gamble2 = list(cs_opt_spec[task]['opt2'].outcomes.values())
# def test_for_ordered_gamble_type(gamble1, gamble2, plot_ordered_gambles_TF=False):
#     ### switched this test to be on the derivatives but finding omegaxy matches up on using utility
#     # uplay_greater_ufold = [calc_CRRA_utility(outcomes=gamble1, omega=w / 100) > calc_CRRA_utility(outcomes=gamble2, omega=w / 100) for w in range(0, 1000)]
#     vplay_greater_vfold = [calc_CRRA_vutil(outcomes=gamble1, omega=w / 100) > calc_CRRA_vutil(outcomes=gamble2, omega=w / 100) for w in
#                            range(0, 1000) if w/100 != 1]
#     num_cross_points = sum([vplay_greater_vfold[i + 1] != vplay_greater_vfold[i] for i in range(len(vplay_greater_vfold) - 1)])
#     loc_cross_points = [[w / 100 for w in range(0, 1000)][i] for i in range(len(vplay_greater_vfold) - 1) if
#                         vplay_greater_vfold[i + 1] != vplay_greater_vfold[i]]
#
#     # plot ordered gamble utilities
#     if plot_ordered_gambles_TF:
#         if (sum(vplay_greater_vfold) != 0) and (sum(vplay_greater_vfold) != len(vplay_greater_vfold)):
#             plt.figure()
#             plt.plot([w / 100 for w in range(0, 1000)],
#                      [calc_CRRA_utility(outcomes=gamble1,
#                                         omega=w / 100) for w in range(0, 1000)])
#             plt.plot([w / 100 for w in range(0, 1000)],
#                      [calc_CRRA_utility(outcomes=gamble2,
#                                         omega=w / 100) for w in range(0, 1000)])
#             plt.title(
#                 'num. cross points: %d at w=%s' % (num_cross_points, loc_cross_points))
#             plt.show()
#
#     if num_cross_points >= 1:
#         omegaxy = [calc_CRRA_utility_omegaxy(gamble1, gamble2, init_guess=loc_cross_points[i]) for i in
#                    range(len(loc_cross_points))]
#         omegaxy = [x for x in omegaxy if x is not None]
#         omegaxy = list(set([round(x * 10000, 0)/10000 for x in omegaxy])) # get unique points to the level of the 3rd decimal place of precision
#         if len(omegaxy) >= 1:
#             Uxy = [(calc_CRRA_utility(outcomes=gamble1, omega=w), calc_CRRA_utility(outcomes=gamble2, omega=w)) for w in omegaxy]
#             Vxy = [(calc_CRRA_vutil(outcomes=gamble1, omega=w), calc_CRRA_vutil(outcomes=gamble2, omega=w)) for w in omegaxy]
#             num_cross_points = len(omegaxy)
#             if num_cross_points == 1:
#                 gamble_type = 'ordered'
#             else:
#                 gamble_type = 'multiple_intersections'
#         else:
#             gamble_type = 'unknown_intersections'
#             num_cross_points = None
#             omegaxy = None
#             Uxy = None
#             Vxy = None
#     elif num_cross_points == 0:
#         gamble_type = 'dominant'
#         omegaxy = None
#         Uxy = None
#         Vxy = None
#     else:
#         return "error in test_for_ordered_gamble_type function"
#
#     return {'type': gamble_type, 'cross points': omegaxy, 'utility_cross': Uxy, 'utility_deriv_cross': Vxy}
#
#     # if num_cross_points == 1:
#     #     return {'type': "ordered", 'cross points': omegaxy, 'utility_cross': Uxy, 'utility_deriv_cross': Vxy}
#     # elif num_cross_points == 0:
#     #     return {'type': "dominant", 'cross points': omegaxy, 'utility_cross': Uxy, 'utility_deriv_cross': Vxy}
#     # elif num_cross_points > 1:
#     #     return {'type': "multiple intersections", 'cross points': omegaxy, 'utility_cross': Uxy, 'utility_deriv_cross': Vxy}
#     # elif num_cross_points == 'unknown':
#     #     return {'type': "unknown intersections", 'cross points': omegaxy, 'utility_cross': Uxy, 'utility_deriv_cross': Vxy}
#     # else:
#     #     return "error in test_for_ordered_gamble_type function"

# def calc_RPM_prob(riskier_TF, omegaxy, omega, kappa, lam, CRRA_order_type):
#     print('WARNING: Needs to be audited. Could not match Apesteguia 2018'
#           'Appendix for all values of p=0.1-0.9 and tasks 1 -4')
#     if CRRA_order_type == 'ordered':
#         if riskier_TF:
#             return calc_RUM_prob(omegaxy, [omegaxy, omega], lam, kappa)
#         else:
#             return calc_RUM_prob(omega, [omegaxy, omega], lam, kappa)
#     elif CRRA_order_type == 'dominant':
#         if riskier_TF:
#             return 1 - kappa
#         else:
#             return kappa
#     else:
#         print('Error calculating calc_RPM_prob: CRRA_order_type is not one of allowable types')
#         return None


# from matplotlib import cm
# from matplotlib import pyplot as plt
# for omega in [(x + 1)/10 for x in range(1, 15)]:
#     fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#     # fig, ax = plt.subplots(1, 1)
#     X = np.arange(0, 1, 0.01)
#     Y = np.arange(0, 100, 1)
#     X, Y = np.meshgrid(X, Y)
#     Z = copy.deepcopy(X)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             Z[i, j] = test.negLL_RUM([X[i, j], Y[i, j], omega])
#     Z[np.isnan(Z)] = 0
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     # plt.imshow(Z, cmap=cm.coolwarm)
#     plt.title('omega = %3.2f' % omega)

# for omega in [(x + 1)/10 for x in range(8, 15)]:
#     fig = plt.figure()
#     plt.plot([x / 100 for x in range(100)],
#              [test.negLL_RUM([kappa, lambda_actual, omega]) for kappa in [x / 100 for x in range(100)]])
#     plt.title('omega=%3.2f, lambda=%3.2f' % (omega, lambda_actual))
#     plt.xlabel('kappa')