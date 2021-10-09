import pickle
import os
import json
import copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, fsolve
import binascii
import random
import math

import matplotlib.pyplot as plt

from assumption_calc_functions import calc_prob_winning_slansky_rank
from assumption_calc_functions import create_game_hand_index
from classify_gambles import identify_omega_range
from classify_gambles import classify_gamble_types

pd.options.display.max_columns = 25

##### take out "perfect_prob"

# ----- File I/O params -----
fp_output = 'output'
fp_choice_situations = os.path.join('choice_situations')
fn_payoff_dict = 'payoff_dict_dnn.json'

# ---- Params -----
# --- data selection params
select_prob_source = 'dnn_prob'   # [perfect_prob, perfect_prob_noise, 'dnn_prob', 'perfect_prob_8020', 'perfect_prob5149']
select_player_list = ['Pluribus']   # ['Eddie', 'MrOrange', 'Joe', 'MrBlonde', 'Gogo', 'Bill', 'MrPink', 'ORen', 'MrBlue', 'Budd', 'MrBrown', 'MrWhite', 'Hattori']   #, 'MrPink']  'Eddie', 'MrOrange', 'Joe', 'MrBlonde', 'Gogo', 'Bill', 'MrPink', 'ORen', 'MrBlue', 'Budd', 'MrBrown', 'Pluribus', 'MrWhite', 'Hattori'
select_case = 'post_neutral_or_blind_only'    # 'post_neutral_or_blind_only'  #'post_loss' #'post_loss_excl_blind_only'  # options: post_loss, post_win, post_loss_excl_blind_only, post_win_excl_blind_only, post_neutral, post_neutral_or_blind_only
select_gamble_types = ['prob_risk_decreases_omega_increases']
fraction_of_data_to_use_for_estimation = 0.8

# ---- pre-calcs ------
fn_prob_dict = {'dnn_prob': 'prob_dict_dnn.json', 'perfect_prob': 'pred_dict_perfect.json', 'perfect_prob_noise': 'pred_dict_perfect_noise.json', 'perfect_prob_8020': 'pred_dict_8020.json', 'perfect_prob_5149': 'pred_dict_5149.json'}[select_prob_source]
select_player_list_save_string = '_'.join(select_player_list).lower()
choice_situations_dir_save_string = os.path.join(fp_output, fp_choice_situations, select_prob_source)
multi_start_dir_save_string = os.path.join('output', 'iter_multistart_saves', select_player_list_save_string, select_prob_source, select_case)
param_estimates_dir_save_string = os.path.join('output', 'iter_multistart_saves', select_player_list_save_string, select_prob_source, 'est_params')

# ---- multi start params
num_multistarts = 1000
save_TF = True
save_iter = 100
t_save_index_start = 200

# ----- Calcs -----
if not save_TF:
    save_iter = True   # pass the false condition on saving to multifactor function

# ----- LOAD DATA -----
# game data
with open("python_hand_data.pickle", 'rb') as f:
    data = pickle.load(f)
players = data['players']
games = data['games']

try:
    with open(os.path.join(fp_output, fn_prob_dict), 'r') as f:
        prob_dict = json.load(f)
    with open(os.path.join(fp_output, fn_payoff_dict), 'r') as f:
        payoff_dict = json.load(f)
except FileNotFoundError:
    print('No probability and payoff dictionaries saved down in output folder')
    prob_dict, payoff_dict = calc_prob_winning_slansky_rank(games, slansky_groups_f=None, seat_groups_f=None, stack_groups_f=None)

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
            if l_dict['gamble_type'] in ['ERROR', 'prob_risk_increases_omega_increases']:
                print('CHECK: %s for game %s hand %s, lambda=%f' % (l_dict['gamble_type'], self.tags['game'], self.tags['hand'], l_level))
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
    # Class ChoiceSituaiton accepts additional specification of "ordered" or "dominant" gamble type,
    # search over range of omegas to determine approximate bounds for assymptotic behavior to 100/0% and 50%

    # currently do not have ordered vs. dominant type working

    def plot_prob(self, add_labels=True):
        for l, t_data in self.CRRA_ordered_gamble_info.items():
            t_vec = [x - t_data['prob_util']['prob'][i - 1] for i, x in enumerate(t_data['prob_util']['prob'])]
            if (t_data['gamble_type'] == 'prob_risk_decreases_omega_increases') and (not ((not all([y > 0 for y in t_vec])) and (not all([y < 0 for y in t_vec])) and (any([y > 0.5 for y in t_data['prob_util']['prob']])))):
                print('WARNING: game %s hand %s classified as %s but some conditions are not met for this classification' %
                      (self.tags['game'], self.tags['hand'], t_type))
            try:
                # plt.figure()    #####
                # plt.title('%s' % t_ind) #######
                t_omega_range = [x / 10000 for x in range(int(t_data['min_omega'] * 10000), int(t_data['max_omega'] * 10000),
                    int((t_data['max_omega'] - t_data['min_omega']) / (len(t_data['prob_util']['prob'])) * 10000))]
                plt.plot(t_omega_range[0:len(t_data['prob_util']['prob'])], t_data['prob_util']['prob'])
                plt.xlabel('omega')
                plt.ylabel('probability of choosing risker gamble, play')
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
    def __init__(self, data_f, param_names_f=['kappa', 'lambda', 'omega']):
        self.data_f = data_f
        self.kappa_f = None
        self.lambda_f = None
        self.omega_f = None
        self.param_names_f = param_names_f   # ['kappa', 'lambda', 'omega']  ###################
        self.init_params = None
        self.results = None

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

    def fit(self, init_params=None, LL_form='RUM', **kwargs):
        if init_params is None:
            self.init_params = [0.5] * len(self.param_names_f)
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
                                                 tags={'game': game_num, 'hand': hand_num, 'player': player_f.name}
                                                 )
            choice_situations_f.append(t_choice_situation)
            del t_choice_situation
            del  tfold_win_prob, tfold_lose_payoff, tplay_win_payoff, tplay_lose_payoff, t_choice_options
            del t_post_loss_bool, t_post_win_bool, t_neutral_bool, t_post_loss_xonlyblind_previous_bool, t_post_win_outcome_xonlyblind_previous_bool, t_neutral_xonlyblind_bool
    del game_num, hands, hand_num
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
    ##### OLD FORMAT #####
    # choice_param_dictionary_f = {int(rank): {int(seat): {'params': dict(), 'n_chosen': {'play': 0, 'fold': 0}, 'CRRA_gamble_type': None, 'CRRA_risky_gamble': None} for seat in set([cs.seat for cs in choice_situations_f])} for rank in set([cs.slansky_strength for cs in choice_situations_f])}
    # for cs in choice_situations_f:
    #     for i in range(len(cs.option_names)):
    #         choice_param_dictionary_f[int(cs.slansky_strength)][int(cs.seat)]['params'].update(
    #             {cs.option_names[i]: list(cs.options[i].outcomes.values())})
    #     choice_param_dictionary_f[int(cs.slansky_strength)][int(cs.seat)]['n_chosen'][cs.choice] += 1
    ################

    choice_param_dictionary_f = {int(rank): {int(seat): [] for seat in set([cs.seat for cs in choice_situations_f])} for
                                 rank in set([cs.slansky_strength for cs in choice_situations_f])}
    for cs in choice_situations_f:
        t_dict = {'params': {}, 'n_chosen': {'play': 0, 'fold': 0}, 'CRRA_gamble_type': None, 'omega_indifference': None}
        for i in range(len(cs.option_names)):
            t_dict['params'].update({cs.option_names[i]: list(cs.options[i].outcomes.values())})
        t_dict['n_chosen'][cs.choice] += 1
        t_dict['CRRA_gamble_type'] = cs.CRRA_ordered_gamble_type
        t_dict['omega_indifference'] = cs.omega_indifference
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


def print_auditing_calcs(t_choice_param_dictionary, t_params):
    for k, v in t_params.items():
        print('Param %s: %s' % (k, v))
    for task in ['cs1', 'cs2', 'cs3', 'cs4']:
        for p in range(1, 11):
            print('TASK %s for p=%3.2f' % (task, p / 10))
            print('Actual choice:   %3.2f' % (t_choice_param_dictionary[int(task.strip('cs'))][p][0]['n_chosen']['play'] / (
                    t_choice_param_dictionary[int(task.strip('cs'))][p][0]['n_chosen']['play'] +
                    t_choice_param_dictionary[int(task.strip('cs'))][p][0]['n_chosen']['fold'])))
            print('Probability RUM:     %3.2f' % calc_RUM_prob(
                calc_CRRA_utility(t_choice_param_dictionary[int(task.strip('cs'))][p][0]['params']['play'], t_params['omega']),
                [calc_CRRA_utility(t_choice_param_dictionary[int(task.strip('cs'))][p][0]['params'][opt], t_params['omega']) for
                 opt in ['play', 'fold']], t_params['lambda'], t_params['kappa']))
            print('log-likelihood RUM: %3.2f' % - RandomUtilityModel(
                {int(task.strip('cs')): {p: t_choice_param_dictionary[int(task.strip('cs'))][p]}}).negLL_RUM(
                [t_params['kappa'], t_params['lambda'], t_params['omega']]))  # / t_total_choices
            print('\n')


def run_multistart(nstart_points, t_lb, t_ub, model_object, save_iter=False, save_path=os.getcwd(), save_index_start=0):
    initial_points = {'kappa': np.random.uniform(low=t_lb['kappa'], high=t_ub['kappa'], size=nstart_points),
                      'lambda': np.random.uniform(low=t_lb['lambda'], high=t_ub['lambda'], size=nstart_points),
                      'omega': np.random.uniform(low=t_lb['omega'], high=t_ub['omega'], size=nstart_points)}

    # [model_object.negLL_RUM([initial_points[j][i] for j in model_object.param_names_f]) for i in range(nstart_points)]

    # [model_object.negLL_RUM([initial_points[model_object.param_names_f[0]][i], initial_points[model_object.param_names_f[1]][i],
    #                  initial_points[model_object.param_names_f[2]][i]]) for i in range(len(initial_points['kappa']))]

    results_list = list()
    for i in range(len(initial_points['kappa'])):
        if (i % 50) == 0:
            print('Optimizing for starting point %d' % i)
        model_object.fit(init_params=[initial_points[j][i] for j in model_object.param_names_f], #[initial_points[model_object.param_names_f[0]][i],
                                      # initial_points[model_object.param_names_f[1]][i],
                                      # initial_points[model_object.param_names_f[2]][i]],
                         LL_form='RUM',
                         method='l-bfgs-b',
                         bounds=Bounds(lb=[t_lb[model_object.param_names_f[i]] for i in range(len(model_object.param_names_f))], ub=[t_ub[model_object.param_names_f[i]] for i in range(len(model_object.param_names_f))]),
                         options={'disp': False, 'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-5})   # tol=1e-8,
        results_list.append({'initial_point': {j: initial_points[j][i] for j in model_object.param_names_f},
                             'results': copy.deepcopy(model_object.results)})

        # {model_object.param_names_f[0]: initial_points[model_object.param_names_f[0]][i],
        #         model_object.param_names_f[1]: initial_points[model_object.param_names_f[1]][i],
        #         model_object.param_names_f[2]: initial_points[model_object.param_names_f[2]][i]},

        if save_iter is not False:
            if (((i + 1) % save_iter) == 0) and (i > 0):
                try:
                    with open(os.path.join(multi_start_dir_save_string,
                                           'multistart_results_iter' + str(i + save_index_start + 1 - save_iter) + 't' + str(save_index_start + i)), 'wb') as ff:
                        pickle.dump(results_list, ff)  # pickle.dump(results_list[(i + 1 - save_iter):(i + 1)], ff)
                        results_list = list()
                except FileNotFoundError:
                    try:
                        os.makedirs(multi_start_dir_save_string)
                        with open(os.path.join(multi_start_dir_save_string,
                                               'multistart_results_iter' + str(
                                                   i + save_index_start + 1 - save_iter) + 't' + str(
                                                   save_index_start + i)), 'wb') as ff:
                            pickle.dump(results_list, ff)  # pickle.dump(results_list[(i + 1 - save_iter):(i + 1)], ff)
                            results_list = list()
                    except FileExistsError as e:
                        print(e)

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


# ====== Import data ======
# --- Can either import saved data or generate again ---
# ---- Calculations -----
try:
    with open(os.path.join(choice_situations_dir_save_string, select_player_list_save_string), 'rb') as f:
        choice_situations = pickle.load(f)
    print('Imported saved choice situations for %s' % select_player_list)
except FileNotFoundError:
    print('No saved choice situations found for player %s, creating now' % select_player_list)
    choice_situations = list()
    for select_player in select_player_list:
        select_player_index = [i for i in range(0, len(players)) if players[i].name == select_player][0]
        game_hand_player_index = create_game_hand_index(players[select_player_index])

        # ---- actual data ----
        choice_situations.append(generate_choice_situations(player_f=players[select_player_index], payoff_dict_f=payoff_dict, prob_dict_f=prob_dict))
    del select_player
    choice_situations = [cs for p in choice_situations for cs in p]
    # add gamble type evaluations
    for cs in choice_situations:
        cs.CRRA_ordered_gamble_info = cs.get_gamble_info(lambda_list_f=[0.5, 1, 2, 10])
        cs.evaluate_gamble_info()
        cs.find_omega_indifference()

    if save_TF:
        try:
            with open(os.path.join(choice_situations_dir_save_string, select_player_list_save_string), 'wb') as f:
                pickle.dump(choice_situations, f)
        except FileNotFoundError:
            try:
                os.makedirs(os.path.join(choice_situations_dir_save_string))
                with open(os.path.join(choice_situations_dir_save_string, select_player_list_save_string), 'wb') as f:
                    pickle.dump(choice_situations, f)
            except FileExistsError as e:
                print(e)


# ------ INSPECT TAGGED GAMBLES BY TYPE (CAN DELETE) --------
# --- collect data
choice_situations_by_case = {'all': choice_situations, 'post loss': copy.deepcopy([cs for cs in choice_situations if cs.__getattribute__('post_loss_excl_blind_only')]), 'post neutral': copy.deepcopy([cs for cs in choice_situations if cs.__getattribute__('post_neutral_or_blind_only')])}
gamble_type_summary = dict()
actual_play_type_summary = dict()
for t_case_name, t_select_choice_situations in choice_situations_by_case.items():
    gamble_type_summary[t_case_name] = {l: {t: list() for t in set([x.CRRA_ordered_gamble_info[l]['gamble_type'] for i, x in enumerate(t_select_choice_situations) if x.CRRA_ordered_gamble_info is not None])}
                           for l in list(set([l for x in t_select_choice_situations if x.CRRA_ordered_gamble_info is not None for l in x.CRRA_ordered_gamble_info.keys()]))}
    actual_play_type_summary[t_case_name] = {l: {t: list() for t in set([x.CRRA_ordered_gamble_info[l]['gamble_type'] for i, x in enumerate(t_select_choice_situations) if x.CRRA_ordered_gamble_info is not None])}
                                for l in list(set([l for x in t_select_choice_situations if x.CRRA_ordered_gamble_info is not None for l in x.CRRA_ordered_gamble_info.keys()]))}
    for l in list(set([l for x in t_select_choice_situations if x.CRRA_ordered_gamble_info is not None for l in x.CRRA_ordered_gamble_info.keys()])):
        for i, x in enumerate(t_select_choice_situations):
            if x.CRRA_ordered_gamble_info is not None:
                gamble_type_summary[t_case_name][l][x.CRRA_ordered_gamble_info[l]['gamble_type']].append(i)
                if x.choice == 'play':
                    actual_play_type_summary[t_case_name][l][x.CRRA_ordered_gamble_info[l]['gamble_type']].append(i)

# --- print summary
for t_case_name in gamble_type_summary.keys():
    print('\n----- %s -------' % t_case_name)
    for l in gamble_type_summary[t_case_name].keys():
        print('\nFor lambda = %3.2f' % l)
        for k, v in {k: len(v) for k, v in gamble_type_summary[t_case_name][l].items()}.items():
            print('%s: %d (%3.1f%% chose play, n=%d)' % (k, v, len(actual_play_type_summary[t_case_name][l][k])/v*100, len(actual_play_type_summary[t_case_name][l][k])))
        del k, v
    del l
del t_case_name

# -- calc tstats for case differences
from assumption_calc_functions import two_sample_test_prop
case_comparison = [('post neutral', 'post loss')]
for case in case_comparison:
    for l in sorted(actual_play_type_summary[case[0]].keys()):
        print('\n----- For lambda = %3.2f -----' % l)
        for g_type in actual_play_type_summary[case[0]][l].keys():
            try:
                t_p1 = len(actual_play_type_summary[case[0]][l][g_type])/len(gamble_type_summary[case[0]][l][g_type])
                t_n1 = len(gamble_type_summary[case[0]][l][g_type])
                t_p2 = len(actual_play_type_summary[case[1]][l][g_type])/len(gamble_type_summary[case[1]][l][g_type])
                t_n2 = len(gamble_type_summary[case[1]][l][g_type])
                t_prop_test = two_sample_test_prop(t_p1, t_p2, t_n1, t_n2, n_sides_f=2)
                print('(%s) %s, diff in play perc: %3.1f%%, pval = %3.1f%%' % ("YES" if t_prop_test[1] <= 0.05 else "NO", g_type, (t_p1 - t_p2)*100, t_prop_test[1]*100))
                print('---- %s=%3.1f%% (n=%d), %s=%3.1f%% (n=%d)' % (case[0], t_p1 * 100, t_n1, case[1], t_p2 * 100, t_n2))
                del t_p1, t_n1, t_p2, t_n2
            except KeyError:
                print('WARNING: Key error for %s' % g_type)
        del g_type
    del l
del case

# -- plot gamble type probabilities for visual inspection
for case in [{'post_loss_excl_blind_only': 'post loss', 'post_neutral_or_blind_only': 'post neutral', 'all': 'all' }[select_case]]:
    t_select_choice_situations = choice_situations_by_case[case]
    for l in gamble_type_summary[case].keys():
        for t_type in gamble_type_summary[case][l].keys():
            plt.figure()
            plt.title('%s at lambda=%3.2f' % (t_type, l))
            for t_ind in gamble_type_summary[case][l][t_type]:
                t_vec = [x - t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['prob_util']['prob'][i - 1]
                         for i, x in
                         enumerate(t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['prob_util']['prob'])]
                if (t_type == 'prob_risk_decreases_omega_increases') and (not (
                        (not all([y > 0 for y in t_vec])) and (not all([y < 0 for y in t_vec])) and (any(
                        [y > 0.5 for y in
                         t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['prob_util']['prob']])))):
                    print(
                        'WARNING: game %s hand %s classified as %s but some conditions are not met for this classification' % (
                        t_select_choice_situations[t_ind].tags['game'], t_select_choice_situations[t_ind].tags['hand'],
                        t_type))
                try:
                    # plt.figure()    #####
                    # plt.title('%s' % t_ind) #######
                    t_omega_range = [x/10000 for x in range(int(t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['min_omega']*10000),
                          int(t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['max_omega']*10000),
                          int((t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['max_omega'] - t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['min_omega'])/(len(t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['prob_util']['prob']))*10000)
                          )]
                    plt.plot(t_omega_range[0:len(t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['prob_util']['prob'])], t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['prob_util']['prob'])
                    plt.xlabel('omega')
                    plt.ylabel('probability of choosing risker gamble, play')
                except ZeroDivisionError:
                    print('check calc of t_omega range for lambda = %s gamble_type = %s' % (l, t_type))
                del t_vec
            del t_ind
        del t_type
    del l
del case

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
    t_candidates = [cs for cs in choice_situations if (cs.__getattribute__(select_case) and (cs.CRRA_ordered_gamble_type in select_gamble_types))]   # any([v['gamble_type'] in select_gamble_types for l, v in cs.CRRA_ordered_gamble_type.items()]))]
else:
    t_candidates = [cs for cs in choice_situations if (cs.CRRA_ordered_gamble_type in select_gamble_types)] #any([v['gamble_type'] in select_gamble_types for l, v in cs.CRRA_ordered_gamble_type.items()])]
choice_param_dictionary = reformat_choice_situations_for_model(random.sample(t_candidates, round(fraction_of_data_to_use_for_estimation * len(t_candidates))))
omega_max_95percentile = np.quantile([cs.CRRA_ordered_gamble_info[l]['max_omega'] for l in t_candidates[0].CRRA_ordered_gamble_info.keys() for cs in t_candidates], 0.95)
del t_candidates

# ---- synthetic test data -----
# kappa_actual = 0.034    # kappa_RPM = 0.051
# lambda_actual = 0.275   # lambda_RPM = 2.495
# omega_actual = 0.661    # omega_RPM = 0.752
#
# choice_param_dictionary = generate_synthetic_data()
# print_auditing_calcs(choice_param_dictionary, t_params={'kappa': None, 'lambda': None, 'omega': 0.661})    #######
# print('Synthetic data dummary: %d total situations, %d play, %d fold' % (sum([len(y) for x in choice_param_dictionary.values() for y in x.values()]),
#                                                                          sum([z['n_chosen']['play'] for x in choice_param_dictionary.values() for y in x.values() for z in y]),
#                                                                          sum([z['n_chosen']['fold'] for x in choice_param_dictionary.values() for y in x.values() for z in y])))

# ====== Run model fitting =======
# --- create model object
model = RandomUtilityModel(choice_param_dictionary, param_names_f=['omega', 'lambda'])

# --- fit one model
model.negLL_RUM([.034, 0.275, 0.661])
# model.negLL_RPM([.034, 0.275, 0.661])
model.negLL_RUM([.1, 2, 0.5])
lb = {'kappa': 0.000, 'lambda': 0.0000, 'omega': 0}
ub = {'kappa': 0.25, 'lambda': 100, 'omega': 2}  # omega_max_95percentile
# model.kappa_f = kappa_actual    ######
model.fit(init_params=[0.5, 0.5], LL_form='RUM',
         method='l-bfgs-b',
         bounds=Bounds(lb=[lb[model.param_names_f[i]] for i in range(len(model.param_names_f))], ub=[ub[model.param_names_f[i]] for i in range(len(model.param_names_f))]),
         tol=1e-12,
         options={'disp': True, 'maxiter': 500})
model.print()


# ============== EXAMINING LIKELIHOOD, CAN DELETE ====================
def check_likelihood(model_f, lb_f, ub_f, prob_type_f="RUM"):
    # fixomega = 1.01    #model_f.results.x[0]
    # fixlambda = model_f.results.x[1]
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

check_likelihood(model, lb, ub, prob_type_f='RPM')



# # observe likelihood over range of omega for single observation holding lambda fixed
# def truncate_choice_situations(rank_to_keep_f, seat_to_keep_f, max_obs_from_rank_seat_f):
#     t_cpd = copy.deepcopy(choice_param_dictionary)
#     # rank_pop_list = [s for s in t_cpd if s not in [rank_to_keep_f]]
#     # seat_pop_list = list(set([r for _, v in t_cpd.items() for r in list(v.keys()) if r not in [seat_to_keep_f]]))
#     [t_cpd.pop(k) for k in [s for s in t_cpd if s not in [rank_to_keep_f]] if k in list(t_cpd.keys())]
#     [v.pop(k) for k in list(set([r for _, v in t_cpd.items() for r in list(v.keys()) if r not in [seat_to_keep_f]])) for v in t_cpd.values() if k in list(v.keys())]
#     for r in t_cpd.keys():
#         for s in t_cpd[r].keys():
#             t_cpd[r][s] = t_cpd[r][s][0:min(len(t_cpd[r][s]), max_obs_from_rank_seat_f)]
#     del r, s
#     return t_cpd
#
# # test_specs = {9: {3: [{'params': {'play': [{'payoff': 1, 'prob': 0.9}, {'payoff': 60, 'prob': 0.1}], 'fold': [{'payoff': 5, 'prob': 1}, {'payoff': 0, 'prob': 0}]}, 'n_chosen': {'play': 1, 'fold': 0}, 'CRRA_gamble_type': None, 'CRRA_risky_gamble': None}]}}
# # test_specs = {9: {3: [{'params': {'play': [{'payoff': 3850, 'prob': 0.6}, {'payoff': 100, 'prob': 0.4}], 'fold': [{'payoff': 2000, 'prob': 0.6}, {'payoff': 1600, 'prob': 0.4}]}, 'n_chosen': {'play': 0, 'fold': 1}, 'CRRA_gamble_type': None, 'CRRA_risky_gamble': None}]}}
# test_specs = truncate_choice_situations(rank_to_keep_f=9, seat_to_keep_f=3, max_obs_from_rank_seat_f=1)
# model_test = RandomUtilityModel(test_specs, param_names_f=['omega', 'lambda'])
# # model_test.negLL_RUM(list(model.results.x))
# # np.exp(-model_test.negLL_RUM(params))
#
#
# div_factor = 100
# num_points = 40
# lb_f = {'kappa': 0.0, 'lambda': 0, 'omega': 0}
# ub_f = {'kappa': 0.5, 'lambda': 10, 'omega': 10}
#     # lambda_range = [l / div_factor for l in range(int(lb_f['lambda']*div_factor), int(ub_f['lambda']*div_factor), int(div_factor/num_points))] # [l / div_factor for l in range(0, 200, 10)]
# # lambda_range = [o / div_factor for o in range(int(lb_f['lambda'] * div_factor), int(ub_f['lambda'] * div_factor),
# #                                               int((int(ub_f['lambda']*div_factor) - int(lb_f['lambda']*div_factor)) / num_points))]
# lambda_range = [0.5, 1.5, 2.5]
# omega_range = [o / div_factor for o in range(int(lb_f['omega']*div_factor), int(ub_f['omega']*div_factor), int(int(ub_f['omega']*div_factor - int(lb_f['omega']*div_factor)) / num_points))]
#
# test_LL_fixlambda = list()
# for fl in lambda_range:
#     test_LL_fixlambda.append([np.exp(-model_test.negLL_RUM([o, fl])) for o in omega_range])
#
# plt.figure()
# for s in test_LL_fixlambda:
#     plt.plot(omega_range, s)
# plt.title('Negative LL holding lambda fixed at various vals')  # estimate %3.2f' % fixlambda)
# plt.xlabel('omega')
# plt.legend(['lam=' + str(round(l, 2)) for l in lambda_range])

# =============================================

# --- run multistart
# run and save

select_results = run_multistart(nstart_points=num_multistarts, t_lb=lb, t_ub=ub, model_object=model, save_iter=save_iter, save_path=multi_start_dir_save_string, save_index_start=t_save_index_start)

try:
    # load from saved
    select_results = list()
    for fn in [f for f in os.listdir(multi_start_dir_save_string) if os.path.isfile(os.path.join(multi_start_dir_save_string, f)) if f[0] != '.']:  # os.listdir(save_path):
        with open(os.path.join(multi_start_dir_save_string, fn), 'rb') as f:
            select_results = select_results + pickle.load(f)
except FileNotFoundError as e:
    print('No available results with specified configuration\n%s' % e)

#########
# ---- debugging / auditing, can delete
i = 0
eps = 1e-4
for r in select_results:
    if all([(r['results'].x[model.param_names_f.index(j)] > (lb[j] + eps)) & (r['results'].x[model.param_names_f.index(j)] < (ub[j] -eps)) for j in model.param_names_f]) & \
            all([x > 1.96 for x in calc_mle_tstat(r['results'].x, calc_robust_varcov(r['results'].jac, r['results'].hess_inv.todense()))['tstat']]):    # & \
            # (r['results'].message == b'CONVERGENCE: NORM_OF_PROJECTED_RADIENT_<=_PGTOL'):
        print('%d %s' % (i, r['results'].x))
    i += 1
#########

list_dict_params, list_dict_obs = parse_multistart(select_results, param_locs_names_f={k: model.param_names_f.index(k) for k in model.param_names_f}, choice_param_dictionary_f=choice_param_dictionary)
# list_dict_params, list_dict_obs = parse_multistart(select_results, kappa_index=0, lambda_index=model.param_names_f.index('lambda'), omega_index=model.param_names_f.index('omega'), choice_param_dictionary_f=choice_param_dictionary)


########### - CAN DELETE CHECKING RESULTS ################
plt.figure()
plt.hist([x['omega'] for x in list_dict_params])
plt.title('omega before filtering')

df_params = pd.DataFrame(list_dict_params).set_index('est_run_id')
df_obs = pd.DataFrame(list_dict_obs)
eps = 1e-2
tstat_limit = 1.96
df_opt_params = model.param_names_f

lb_additional = {'kappa': lb['kappa'], 'lambda': lb['lambda'], 'omega': lb['omega']}
ub_additional = {'kappa': ub['kappa'], 'lambda': ub['lambda'], 'omega': 2.2}    #ub['omega']

t_filter = pd.Series(True, index=df_params.index)
for n in df_opt_params: #
    # print(n)
    filter_bounds = (df_params[n] >= (lb[n] + eps)) & (df_params[n] <= (ub[n] - eps))
    filter_bounds_additional = (df_params[n] >= (lb_additional[n] + eps)) & (df_params[n] <= (ub_additional[n] - eps))
    filter_tstat = (df_params[n + '_tstat'].abs() >= tstat_limit)
    filter_initial = (df_params[n] >= (df_params[n + '_initial'] + eps)) | (df_params[n] <= (df_params[n + '_initial'] - eps))
    t_filter = t_filter & filter_bounds & filter_bounds_additional & filter_initial & filter_tstat
    print('After %s, %d obs remaining' % (n, sum(t_filter)))
if 'lambda' in df_opt_params:
    filter_lambda1 = (df_params['lambda'] > (1 + eps)) | (df_params['lambda'] < (1 - eps))
else:
    filter_lambda1 = pd.Series(True, index=df_params.index)
if 'omega' in df_opt_params:
    filter_omega1 = (df_params['omega'] > (1 + eps)) | (df_params['omega'] < (1 - eps))

filter_message = ((df_params.message == b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'))# | (df_params.message == b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'))
ind_filter = df_params.index[t_filter & filter_omega1 & filter_lambda1 & filter_message]  #
print('Across all filters %d/%d observations remain' % (len(ind_filter), df_params.shape[0]))

for n in df_opt_params:
    plt.figure()
    df_params.loc[ind_filter][n].hist()
    plt.title('Histogram of ' + n + ' estimates')

# if 'lambda' in df_params.columns:
#     plt.figure()
#     plt.hist([np.log(x) for x in df_params.loc[ind_filter]['lambda']])
#     plt.title('Histogram of ln(lambda) estimates')

df_obs_filtered = df_obs.loc[df_obs.est_run_id.isin(list(ind_filter))]
plt.figure()
# plt.scatter(df_obs_filtered['util_fold'], df_obs_filtered['util_play'], c=df_obs_filtered.actual_share)
plt.scatter(df_obs_filtered.loc[df_obs_filtered.actual_share == 0, 'util_fold'],
            df_obs_filtered.loc[df_obs_filtered.actual_share == 0, 'util_play'], c='r')
plt.scatter(df_obs_filtered.loc[df_obs_filtered.actual_share == 1, 'util_fold'],
            df_obs_filtered.loc[df_obs_filtered.actual_share == 1, 'util_play'], c='b')
plt.legend(['actually folded', 'actually played'])
plt.plot([min(min(df_obs_filtered['util_fold']), min(df_obs_filtered['util_play'])), max(max(df_obs_filtered['util_fold']), max(df_obs_filtered['util_play']))],
         [min(min(df_obs_filtered['util_fold']), min(df_obs_filtered['util_play'])), max(max(df_obs_filtered['util_fold']), max(df_obs_filtered['util_play']))], '.-')
plt.xlabel('Utility of folding at estimated parameters')
plt.ylabel('Utility of playing at estimated parameters')

plt.figure()
plt.scatter(df_obs_filtered['pred_share'], df_obs_filtered['actual_share'])
plt.xlabel('Predicted share')
plt.ylabel('Choice')
plt.title('Actual v. predicted shares')

print('---- AFTER FILTERING FOR VALID CONVERGENCE-------')
print('n obs = %d' % len(ind_filter))
print('mean OMEGA = %3.2f' % df_params.loc[ind_filter]['omega'].mean())
print('mean LAMBDA = %3.2f' % df_params.loc[ind_filter]['lambda'].mean())
print('stdev OMEGA = %3.5f' % df_params.loc[ind_filter]['omega'].std())
print('stdev LAMBDA = %3.5f' % df_params.loc[ind_filter]['lambda'].std())

if 'kappa' in model.param_names_f:
    print('mean KAPPA = %3.2f' % df_params.loc[ind_filter]['kappa'].mean())
    print('stdev KAPPA = %3.5f' % df_params.loc[ind_filter]['kappa'].std())


# df_obs_filtered['pred_exp_value'] = df_obs_filtered.pred
exp_omega = df_params.loc[ind_filter]['omega'].mean()
for rank in choice_param_dictionary.keys():
    for seat in choice_param_dictionary[rank].keys():
        for item in choice_param_dictionary[rank][seat]:
            item.update({'exp_util_omega_' + str(int(round(exp_omega, 4)*10000)) + 'e4': {'play': calc_CRRA_utility(item['params']['play'], exp_omega),
                                                                                          'fold': calc_CRRA_utility(item['params']['fold'], exp_omega)}})

df_rational = pd.DataFrame(columns=['play_TF', 'fold_TF', 'exp_util_play', 'exp_util_fold'])
for rank in choice_param_dictionary.keys():
    for seat in choice_param_dictionary[rank].keys():
        for item in choice_param_dictionary[rank][seat]:
            t_key = [k for k in item.keys() if k.find('exp_util_omega_') > -1][0]
            df_rational = df_rational.append(
                pd.Series({'play_TF': bool(item['n_chosen']['play']),
                           'fold_TF': bool(item['n_chosen']['fold']),
                           'exp_util_play': item[t_key]['play'],
                           'exp_util_fold': item[t_key]['fold']}),
                ignore_index=True)
            del t_key

df_rational = df_rational.astype({'play_TF': bool, 'fold_TF': bool})
df_rational['exp_util_play_greater_than_fold'] = (df_rational.exp_util_play > df_rational.exp_util_fold)
df_rational['rational'] = (df_rational.exp_util_play_greater_than_fold & df_rational.play_TF) | ((~df_rational.exp_util_play_greater_than_fold) & (~df_rational.play_TF))


print('--- Rational choice:')
print('(Rational) Blended: %3.1f%% (n=%d):' % (df_rational.rational.sum() / df_rational.shape[0] * 100, df_rational.shape[0]))
print('(Rational)   Play - positive expected value: %3.1f%% (n=%d)' % (sum(df_rational.exp_util_play_greater_than_fold & df_rational.play_TF)/sum(df_rational.exp_util_play_greater_than_fold) * 100, sum(df_rational.exp_util_play_greater_than_fold & df_rational.play_TF)))
print('(Irrational) Fold - positive expected value: %3.1f%% (n=%d)' % (sum(df_rational.exp_util_play_greater_than_fold & ~df_rational.play_TF)/sum(df_rational.exp_util_play_greater_than_fold) * 100, sum(df_rational.exp_util_play_greater_than_fold & ~df_rational.play_TF)))
print('(Irrational)   Play - negative expected value: %3.1f%% (n=%d)' % (sum(~df_rational.exp_util_play_greater_than_fold & df_rational.play_TF)/sum(~df_rational.exp_util_play_greater_than_fold) * 100, sum(~df_rational.exp_util_play_greater_than_fold & df_rational.play_TF)))
print('(Rational) Fold - negative expected value: %3.1f%% (n=%d)' % (sum(~df_rational.exp_util_play_greater_than_fold & ~df_rational.play_TF)/sum(~df_rational.exp_util_play_greater_than_fold) * 100, sum(~df_rational.exp_util_play_greater_than_fold & ~df_rational.play_TF)))


#################################

# ====== SAVE TO CSV FOR TABLEAU EXAMINATION =======
# select_case_hold = select_case  #####
# select_case = select_case_hold + '_30perc_draw_test'    ######
# if save_TF:
#     pd.DataFrame(list_dict_params).set_index('est_run_id').to_csv(os.path.join('output', select_player_list_save_string + '_multistart_params_' + select_case + '.csv'))
#     pd.DataFrame(list_dict_obs).set_index(['est_run_id', 'rank', 'seat']).to_csv(os.path.join('output', select_player_list_save_string + '_multistart_obs_' + select_case + '.csv'))

# save estimated parameter output
if save_TF:
    try:
        with open(os.path.join(param_estimates_dir_save_string, select_player_list_save_string +'_' + select_case + '.json'), 'w') as f:
            t_dict = {p: {'mean': df_params.loc[ind_filter][p].mean(), 'stdev': df_params.loc[ind_filter][p].std(), 'nobs': float(df_params.loc[ind_filter][p].count())} for p in ['omega', 'lambda']}
            t_dict.update({'proportion_rational': df_rational.rational.sum() / df_rational.shape[0], 'nobs_rational': df_rational.shape[0]})
            json.dump(t_dict, fp=f)
            del t_dict
    except FileNotFoundError:
        try:
            os.makedirs(param_estimates_dir_save_string)
            with open(os.path.join(param_estimates_dir_save_string, select_player_list_save_string +'_' + select_case + '.json'), 'w') as f:
                t_dict = {p: {'mean': df_params.loc[ind_filter][p].mean(), 'stdev': df_params.loc[ind_filter][p].std(), 'nobs': float(df_params.loc[ind_filter][p].count())} for p in ['omega', 'lambda']}
                t_dict.update({'proportion_rational': df_rational.rational.sum() / df_rational.shape[0], 'nobs_rational': df_rational.shape[0]})
                json.dump(t_dict, fp=f)
                del t_dict
        except FileExistsError as e:
            print(e)

# ============== TEST OF DIFFERENCES ==============
from assumption_calc_functions import two_sample_test_ind_means
from assumption_calc_functions import two_sample_test_prop

# specify players and cases
players_for_testing = ['pluribus', 'eddie', 'bill', 'eddie_mrorange_joe_mrblonde_gogo_bill_mrpink_oren_mrblue_budd_mrbrown_mrwhite_hattori'] # 'mrpink', 'mrorange', 'bill',
cases_for_testing = ['post_neutral_or_blind_only', 'post_loss_excl_blind_only', ] # must only be a list of two currently
params_for_testing = ['omega', 'lambda']
prob_case_for_testing = 'dnn_prob'

# load relevant data
results_dict = {p: {c: {} for c in cases_for_testing} for p in players_for_testing}
for p in players_for_testing:
    for c in cases_for_testing:
        with open(os.path.join('output', 'iter_multistart_saves', p, prob_case_for_testing, 'est_params', p + '_' + c + '.json'), 'r') as f:
            results_dict[p][c] = json.load(f)

# compare parameter estimates by case
print('\n\n===== change in RISK =====')
print('For two sample test of independent means (expected value of estimated parameters)')
print('case 1: %s, \t case2: %s' % (cases_for_testing[0], cases_for_testing[1]))
for p in players_for_testing:
    print('\n\n\tFor player %s:' % p)
    for param in params_for_testing:
        print('\t\tParameter: %s' % param)
        print('\t\t\tCase 1: %s, mean = %3.2f, stdev = %3.2f, n = %d' % (cases_for_testing[0], results_dict[p][cases_for_testing[0]][param]['mean'], results_dict[p][cases_for_testing[0]][param]['stdev'], results_dict[p][cases_for_testing[0]][param]['nobs']))
        print('\t\t\tCase 2: %s, mean = %3.2f, stdev = %3.2f, n = %d' % (cases_for_testing[1], results_dict[p][cases_for_testing[1]][param]['mean'], results_dict[p][cases_for_testing[1]][param]['stdev'], results_dict[p][cases_for_testing[1]][param]['nobs']))
        print('\t\tt-stat: %3.2f, p-value: %3.1f' % (two_sample_test_ind_means(results_dict[p][cases_for_testing[0]][param]['mean'], results_dict[p][cases_for_testing[1]][param]['mean'],
                                                                                 results_dict[p][cases_for_testing[0]][param]['stdev'], results_dict[p][cases_for_testing[1]][param]['stdev'],
                                                                                 results_dict[p][cases_for_testing[0]][param]['nobs'], results_dict[p][cases_for_testing[1]][param]['nobs'], n_sides=2, print_f=False)))

# compare proportion of rational decisions by case
print('\n\n===== change in RATIONALITY ======')
print('For two sample test of proportions (change in proportion of rational actions)')
print('case 1: %s, \t case2: %s' % (cases_for_testing[0], cases_for_testing[1]))
for p in players_for_testing:
    print('\tFor player %s:' % p)
    print('\t\tCase 1: %s, proportion = %3.2f, n = %d' % (cases_for_testing[0], results_dict[p][cases_for_testing[0]]['proportion_rational'], results_dict[p][cases_for_testing[0]]['nobs_rational']))
    print('\t\tCase 2: %s, proportion = %3.2f, n = %d' % (cases_for_testing[1], results_dict[p][cases_for_testing[1]]['proportion_rational'], results_dict[p][cases_for_testing[1]]['nobs_rational']))
    print('\tt-stat: %3.2f, p-value: %3.1f' % (two_sample_test_prop(results_dict[p][cases_for_testing[0]]['proportion_rational'], results_dict[p][cases_for_testing[1]]['proportion_rational'],
                                                 results_dict[p][cases_for_testing[0]]['nobs_rational'], results_dict[p][cases_for_testing[1]]['nobs_rational'],
                                                 n_sides_f=2)))
