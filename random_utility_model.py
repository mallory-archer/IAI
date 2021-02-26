import pickle
import os
import json
# import pandas as pd
import numpy as np

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

    # def calc_option_value(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f):
    #     return sum([calc_outcome_value(
    #         calc_outcome_v(x_f=t_outcome['payoff'], alpha_f=alpha_f, lambda_f=lambda_f, beta_f=beta_f),
    #         calc_outcome_pi(p_f=t_outcome['prob'], c_f=gamma_f if (t_outcome['payoff'] >= 0) else delta_f))
    #                 for _, t_outcome in self.outcomes.items()])
    #
    # # def calc_option_utility(self):
    # #     return sum([calc_outcome_value(v_f=t_outcome['payoff'], pi_f=t_outcome['prob']) for _, t_outcome in self.outcomes.items()])
    #
    # def calc_option_utility(self, lose_weight_f=1):
    #     return sum([calc_outcome_value(v_f=t_outcome['payoff'], pi_f=t_outcome['prob']) * (lose_weight_f if t_name == 'lose' else 1) for t_name, t_outcome in self.outcomes.items()])


class ChoiceSituation:
    def __init__(self, sit_options, sit_choice=None, slansky_strength=None, stack_rank=None, seat=None, post_loss=None):
        self.options = sit_options
        self.option_names = [x.name for x in sit_options]
        self.choice = sit_choice
        self.slansky_strength = slansky_strength
        self.stack_rank = stack_rank
        self.seat = seat
        self.post_loss = post_loss
        # self.values = None
        # self.prob_options = None
        # self.prob_choice = None
        # self.pred_choice = None
        # self.util_values = None
        # self.util_prob_options = None
        # self.util_prob_choice = None
        # if (self.choice is not None) and self.choice not in self.option_names:
        #     print("Warning: specified choice of '%s' is not an option in choice set: %s" % (
        #     self.choice, self.option_names))

    def print(self):
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))

    # prospect model formulation
    # def get_option_values(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f):
    #     return [x.calc_option_value(alpha_f, lambda_f, beta_f, gamma_f, delta_f) for x in self.options]
    #
    # def get_option_probs(self, phi_f):
    #     if self.values is not None:
    #         return calc_prob_options(self.values, phi_f)
    #     else:
    #         print("Error: no option values stored for choice situation. Set via method 'get_option_values(args)' first")
    #         return None
    #
    # def get_choice_location(self, choice_f):
    #     return self.option_names.index(choice_f)
    #
    # def get_prob_choice(self, choice_f):
    #     if self.prob_options is not None:
    #         return self.prob_options[self.get_choice_location(choice_f)]
    #     else:
    #         print(
    #             "Error: no option probabilities stored for choice situation. Set via method 'get_option_probs(args)' first")
    #         return None
    #
    # def set_model_values(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f, phi_f):
    #     self.values = self.get_option_values(alpha_f, lambda_f, beta_f, gamma_f, delta_f)
    #     self.prob_options = self.get_option_probs(phi_f)
    #     self.prob_choice = self.get_prob_choice(self.choice)
    #
    # def predict_choice(self):
    #     self.pred_choice = self.option_names[self.prob_options.index(max(self.prob_options))]
    #
    # # traditional utility model formulation
    # def get_util_values(self, lose_weight_f):
    #     return [x.calc_option_utility(lose_weight_f) for x in self.options]
    #
    # def get_util_option_probs(self, phi_f):
    #     if self.util_values is not None:
    #         return calc_prob_options(self.util_values, phi_f)
    #     else:
    #         print("Error: no utility values stored for choice situation. Set via method 'get_option_values(args)' first")
    #         return None
    #
    # def get_util_prob_choice(self, choice_f):
    #     if self.util_prob_options is not None:
    #         return self.util_prob_options[self.get_choice_location(choice_f)]
    #     else:
    #         print(
    #             "Error: no option probabilities stored for choice situation. Set via method 'get_option_probs(args)' first")
    #         return None
    #
    # def set_util_model_values(self, lose_weight_f, phi_f):
    #     self.util_values = self.get_util_values(lose_weight_f)
    #     self.util_prob_options = self.get_util_option_probs(phi_f)
    #     self.util_prob_choice = self.get_util_prob_choice(self.choice)


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
    unique_slansky = set([cs.slansky_strength for cs in choice_situations])
    unique_seat = set([cs.seat for cs in choice_situations])
    choice_param_dictionary = dict(zip(unique_slansky, [
        dict(zip(unique_seat, [{'params': dict(), 'n_chosen': {'play': 0, 'fold': 0}}] * len(unique_seat)))] * len(
        unique_slansky)))
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
            return prob * (payoff ** (1 - omega)) / (1 - omega)
    return sum([calc_outcome_util(o['prob'], o['payoff'], omega) for o in outcomes])


def calc_RUM_LL(kappa, lam, omega, data):
    def calc_LLi(X, Y, I, util_X, util_Y, kappa, lam):
        return (X + I/2) * np.log(((1-2*kappa)*np.exp(lam*util_X)) / (np.exp(lam*util_X + np.exp(lam*util_Y))) + kappa) + (Y + I / 2) * np.log(((1 - 2 * kappa) * np.exp(lam * util_Y)) / (np.exp(lam * util_X + np.exp(lam * util_Y))) + kappa)
    LLi = list()
    for rank in data.keys():
        for seat in data[rank].keys():
            Xi = data[rank][seat]['n_chosen']['play']
            Yi = data[rank][seat]['n_chosen']['fold']
            Ii = 0
            util_Xi = calc_CRRA_utility(data[rank][seat]['params']['play'], omega)
            util_Yi = calc_CRRA_utility(data[rank][seat]['params']['fold'], omega)
            LLi.append(calc_LLi(Xi, Yi, Ii, util_Xi, util_Yi, kappa, lam))
    return sum(LLi)


# Run code
choice_situations = generate_choice_situations(player_f=players[select_player_index], game_hand_index_f=game_hand_player_index, payoff_dict_f=payoff_dict, prob_dict_f=prob_dict)

choice_param_dictionary = reformat_choice_situations_for_model(choice_situations)

LL_RUM = calc_RUM_LL(kappa=0.008, lam=4.572, omega=2, data=choice_param_dictionary)

# ---------- ARCHIVE -------
# data frame for sanity checking / working
# df_choices = pd.DataFrame(columns=['slansky', 'seat', 'choice', 'post_loss'])
# for cs in choice_situations:
#     df_choices = df_choices.append(dict(zip(['slansky', 'seat', 'choice', 'post_loss'], [cs.slansky_strength, cs.seat, cs.choice, cs.post_loss])), ignore_index=True)
