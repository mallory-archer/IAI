import pickle
import os
import json
# import pandas as pd
import numpy as np
from scipy.optimize import minimize

from assumption_calc_functions import calc_prob_winning_slansky_rank
from assumption_calc_functions import create_game_hand_index

# ----- File I/O params -----
fp_output = 'output'
fn_prob_payoff_dict = 'prob_payoff_dicts.json'

# ---- Params -----
select_player = 'Pluribus'

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
    def __init__(self, sit_options, sit_choice=None, slansky_strength=None, stack_rank=None, seat=None, post_loss=None):
        self.options = sit_options
        self.option_names = [x.name for x in sit_options]
        self.choice = sit_choice
        self.slansky_strength = slansky_strength
        self.stack_rank = stack_rank
        self.seat = seat
        self.post_loss = post_loss

    def print(self):
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))


def generate_choice_situations(player_f, game_hand_index_f, prob_dict_f, payoff_dict_f):
    choice_situations_f = list()
    num_choice_situations_dropped = 0

    for game_num, hands in game_hand_index_f.items():
        for hand_num in hands:
            try:
                big_blind = 100
                small_blind = 50
                payoff_units_f = big_blind * 1

                t_slansky_rank = str(player_f.odds[game_num][hand_num]['slansky'])
                t_seat_num = str(player_f.seat_numbers[game_num][hand_num])

                # --- aggregate to rank level
                tplay_win_prob = prob_dict_f[t_slansky_rank][t_seat_num]['win'] / prob_dict_f[t_slansky_rank][t_seat_num]['play_count']
                tplay_win_payoff = payoff_dict_f[t_slansky_rank][t_seat_num]['win_sum'] / payoff_dict_f[t_slansky_rank][t_seat_num]['win_count']
                tplay_lose_payoff = payoff_dict_f[t_slansky_rank][t_seat_num]['loss_sum'] / payoff_dict_f[t_slansky_rank][t_seat_num]['loss_count']

                tfold_win_prob = 0  # cannot win under folding scenario
                if player_f.blinds[game_num][hand_num]['big']:
                    tfold_lose_payoff = (big_blind * -1)/payoff_units_f
                elif player_f.blinds[game_num][hand_num]['small']:
                    tfold_lose_payoff = (small_blind * -1)/payoff_units_f
                else:
                    tfold_lose_payoff = 0/payoff_units_f

                t_choice_options = [Option(name='play', outcomes={'win': {'payoff': tplay_win_payoff, 'prob': tplay_win_prob},
                                                                  'lose': {'payoff': tplay_lose_payoff, 'prob': 1 - tplay_win_prob}}),
                                    Option(name='fold', outcomes={'lose': {'payoff': tfold_lose_payoff, 'prob': 1 - tfold_win_prob}})]
                try:
                    t_post_loss_bool = (player_f.outcomes[game_num][str(int(hand_num)-1)] < 0)
                except KeyError:
                    t_post_loss_bool = None

                t_choice_situation = ChoiceSituation(sit_options=t_choice_options[:],
                                                     sit_choice="fold" if player_f.actions[game_num][hand_num]['preflop'] == 'f' else "play",
                                                     slansky_strength=player_f.odds[game_num][hand_num]['slansky'],
                                                     stack_rank=player_f.stack_ranks[game_num][hand_num],
                                                     seat=player_f.seat_numbers[game_num][hand_num],
                                                     post_loss=t_post_loss_bool)

                choice_situations_f.append(t_choice_situation)

                del t_choice_situation
            except KeyError:
                num_choice_situations_dropped += 1
    del game_num, hands, hand_num, t_slansky_rank, t_seat_num
    print('Dropped a total of %d for KeyErrors \n(likely b/c no observations for combination of slansky/seat/stack for prob and payoff estimates.) \nKept a total of %d choice situations' % (num_choice_situations_dropped, len(choice_situations_f)))
    return choice_situations_f


def reformat_choice_situations_for_model(choice_situations):
    # create dictionary of option params
    choice_param_dictionary = {rank: {seat: {'params': dict(), 'n_chosen': {'play': 0, 'fold': 0}} for seat in set([cs.seat for cs in choice_situations])} for rank in set([cs.slansky_strength for cs in choice_situations])}
    for cs in choice_situations:
        for i in range(len(cs.option_names)):
            choice_param_dictionary[cs.slansky_strength][cs.seat]['params'].update(
                {cs.option_names[i]: list(cs.options[i].outcomes.values())})
        choice_param_dictionary[cs.slansky_strength][cs.seat]['n_chosen'][cs.choice] += 1
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

# def calc_RUM_LL(kappa, lam, omega, data):
#     def calc_LLi(X, Y, I, util_X, util_Y, kappa, lam):
#         return (X + I/2) * np.log(((1-2*kappa)*np.exp(lam*util_X)) / (np.exp(lam*util_X) + np.exp(lam*util_Y)) + kappa) + \
#                (Y + I / 2) * np.log(((1 - 2 * kappa) * np.exp(lam * util_Y)) / (np.exp(lam * util_X) + np.exp(lam * util_Y)) + kappa)
#     LLi = list()
#     for rank in data.keys():
#         for seat in data[rank].keys():
#             # Xi = data[rank][seat]['n_chosen']['play']
#             # Yi = data[rank][seat]['n_chosen']['fold']
#             # Ii = 0
#             # util_Xi = calc_CRRA_utility(data[rank][seat]['params']['play'], omega)
#             # util_Yi = calc_CRRA_utility(data[rank][seat]['params']['fold'], omega)
#             LLi.append(calc_LLi(X=data[rank][seat]['n_chosen']['play'],
#                                 Y=data[rank][seat]['n_chosen']['fold'],
#                                 I=0,
#                                 util_X=calc_CRRA_utility(data[rank][seat]['params']['play'], omega),
#                                 util_Y=calc_CRRA_utility(data[rank][seat]['params']['fold'], omega),
#                                 kappa=kappa,
#                                 lam=lam))
#     return sum(LLi)


# Run code

choice_situations = generate_choice_situations(player_f=players[select_player_index], game_hand_index_f=game_hand_player_index, payoff_dict_f=payoff_dict, prob_dict_f=prob_dict)

choice_param_dictionary = reformat_choice_situations_for_model(choice_situations)

# LL_RUM = calc_RUM_LL(kappa=0.008, lam=4.572, omega=2, data=choice_param_dictionary)


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
    def negLL(self, params):
        def calc_LLi(X, Y, I, util_X, util_Y, kappa, lam):
            return (X + I / 2) * np.log(((1 - 2 * kappa) * np.exp(lam * util_X)) / (
                    np.exp(lam * util_X) + np.exp(lam * util_Y)) + kappa) + \
                   (Y + I / 2) * np.log(((1 - 2 * kappa) * np.exp(lam * util_Y)) / (
                    np.exp(lam * util_X) + np.exp(lam * util_Y)) + kappa)

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

        return -sum(LLi)*10

    def fit(self, init_params=None):
        if init_params is None:
            self.init_params = [1] * len(self.param_names_f)
        else:
            self.init_params = init_params

        self.results = minimize(self.negLL, self.init_params, method="Nelder-Mead", options={"disp": True})

    def print(self):
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))

import copy
test_dict = copy.deepcopy(choice_param_dictionary)
# calc_RUM_LL(kappa=0.008, lam=4.572, omega=2, data=test_dict)
test_dict.pop(9)
test_dict[5].pop(4)

test = RandomUtilityModel(choice_param_dictionary)
test.negLL([0.008, 4.5, 0.5])
test.fit(init_params=[0.008, 5, 0.5])
test.print()

import matplotlib.pyplot as plt
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


plt.scatter([x for x in [i/100 for i in range(1, 300)]], [test_CRRA_calc(choice_param_dictionary[5][6]['params']['play'], x) for x in [i/100 for i in range(1, 300)]])

def calc_prob(outcomes_x, outcomes_y, lambda_f, omega_f):
    Ux = test_CRRA_calc(data=outcomes_x, x=omega_f)
    Uy = test_CRRA_calc(data=outcomes_y, x=omega_f)
    return np.exp(lambda_f * Ux) / (np.exp(lambda_f * Ux) + np.exp(lambda_f * Uy))
outcomes_x = [{'payoff': -1, 'prob': 0.9}, {'payoff': 100, 'prob': 0.1}]
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
