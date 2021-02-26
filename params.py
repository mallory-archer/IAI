import numpy as np
import scipy.optimize
import assumption_calc_functions as acf
import json


def prob_winning_transform(args_list_f, exp_loss_f):
    b_f = args_list_f[0]
    # Infer probability of winning hand based on expected values of winnings collected from online poker site:
    # https: // www.tightpoker.com / poker_hands.html

    # f(x) = b,  f(x) is expected value calculation
    def f(prob_win_ff, payoff_win_ff, payoff_lose_ff):
        return np.array([(p * payoff_win_ff) + ((1 - p) * payoff_lose_ff) for p in prob_win_ff])

    # The second parameter is used to set the solution vector using the args argument of leastsq.
    def system(x_f, b_ff, payoff_lose_f):
        return f(x_f[0:-1], x_f[-1], payoff_lose_f) - b_ff

    result_f = scipy.optimize.least_squares(fun=system, x0=np.array([0.5]*len(b_f) + [3]),
                                            bounds=([1e-4]*len(b_f) + [-100000], [1-1e-4]*len(b_f) + [100000]),
                                            args=(list(b_f.values()), exp_loss_f))
    return dict(zip(list(b_f.keys()), list(result_f['x'][0:-1])))


# --------- CALC PROB WINNING HANDS FROM ONLINE SITE (REQUIRES SENSITIVE ASSUMPTION ABOUT WIN/LOSS AMOUNTS)
# exp_val_hand from https://www.tightpoker.com/poker_hands.html; values in units of big blinds
# exp_val_hand = {'AA': 2.32, 'KK': 1.67, 'QQ': 1.22, 'JJ': 0.86, 'AKs': 0.78, 'AQs': 0.59, 'TT': 0.58, 'AK': 0.51,
#                 'AJs': 0.44, 'KQs': 0.39, '99': 0.38, 'ATs': 0.32, 'AQ': 0.31, 'KJs': 0.29, '88': 0.25, 'QJs': 0.23,
#                 'KTs': 0.2, 'A9s': 0.19, 'AJ': 0.19, 'QTs': 0.17, 'KQ': 0.16, '77': 0.16, 'JTs': 0.15, 'A8s': 0.1,
#                 'K9s': 0.09, 'AT': 0.08, 'A5s': 0.08, 'A7s': 0.08, 'KJ': 0.08, '66': 0.07, 'T9s': 0.05, 'A4s': 0.05,
#                 'Q9s': 0.05, 'J9s': 0.04, 'QJ': 0.03, 'A6s': 0.03, '55': 0.02, 'A3s': 0.02, 'K8s': 0.01, 'KT': 0.01,
#                 '98s': 0, 'T8s': 0, 'K7s': 0, 'A2s': 0, '87s': -0.02, 'QT': -0.02, 'Q8s': -0.02, '44': -0.03,
#                 'A9': -0.03, 'J8s': -0.03, '76s': -0.03, 'JT': -0.03, '97s': -0.04, 'K6s': -0.04, 'K5s': -0.05,
#                 'K4s': -0.05, 'T7s': -0.05}
# run function "prob_winning_transform" from odds_functions once with the above param vector as the arg
# prob_hand_dict = prob_winning_transform([exp_val_hand], -1)


# --------------------- INFER WIN PROBABILITIES AND PAYOFFS FROM DATA SET ---------------------------
# ---- Run summary functions on actual data to approximate win / loss probabilities and payoffs; output of acf function has other options for levels of aggregation
# def create_prob_payoff_save_strings(slansky_groups_f, seat_groups_f, stack_groups_f):
#     slansky_save_string_f = '_' + '_'.join([''.join([x for x in y]) for y in slansky_groups_f]) if slansky_groups_f is not None else ''
#     seat_save_string_f = '_' + '_'.join([''.join([x for x in y]) for y in seat_groups_f]) if seat_groups_f is not None else ''
#     stack_save_string_f = '_' + '_'.join([''.join([x for x in y]) for y in stack_groups_f]) if stack_groups_f is not None else ''
#
#     prob_save_string_f = 'prob_slansky' + slansky_save_string_f + '_seat' + seat_save_string_f + '_stack' + stack_save_string_f
#     payoff_save_string_f = 'payoff_slansky' + slansky_save_string_f + '_seat' + seat_save_string_f + '_stack' + stack_save_string_f
#     return prob_save_string_f, payoff_save_string_f, slansky_save_string_f, seat_save_string_f, stack_save_string_f

# def create_prob_payoff_dict(games_f, slansky_groups_f=None, seat_groups_f=None, stack_groups_f=None, write_to_file_TF_f=False):
#     slansky_groups_f = [[str(x) for x in y] for y in slansky_groups_f] if slansky_groups_f is not None else None
#     seat_groups_f = [[str(x) for x in y] for y in seat_groups_f] if seat_groups_f is not None else None
#     stack_groups_f = [[str(x) for x in y] for y in stack_groups_f] if stack_groups_f is not None else None
#     prob_counts_dict_f, payoff_counts_dict_f, prob_dict_f, payoff_dict_f = acf.calc_prob_winning_slansky_rank(games_f,
#                                                                                                               slansky_groups_f=slansky_groups_f,
#                                                                                                               seat_groups_f=seat_groups_f,
#                                                                                                               stack_groups_f=stack_groups_f
#                                                                                                               )
#     if write_to_file_TF_f:
#         prob_save_string_f, payoff_save_string_f, _, _, _ = create_prob_payoff_save_strings(slansky_groups_f, seat_groups_f, stack_groups_f)
#         with open(prob_save_string_f + '.json','w') as f:
#             json.dump(prob_dict_f, f)
#         with open(payoff_save_string_f + '.json', 'w') as f:
#             json.dump(payoff_dict_f, f)
#
#     return prob_counts_dict_f, payoff_counts_dict_f, prob_dict_f, payoff_dict_f


# def load_prob_payoff_dict(slansky_groups_f=None, seat_groups_f=None, stack_groups_f=None):
#     prob_save_string_f, payoff_save_string_f, _, _, _ = create_prob_payoff_save_strings(slansky_groups_f, seat_groups_f, stack_groups_f)
#     with open(prob_save_string_f + '.json', 'r') as f:
#         prob_dict_f = json.load(f)
#     with open(payoff_save_string_f + '.json', 'r') as f:
#         payoff_dict_f = json.load(f)
#     return prob_dict_f, payoff_dict_f


# ---- Load previously calculated saved probabilities and payoffs ---
# slansky_save_string = '_' + '_'.join([''.join([x for x in y]) for y in slansky_groups]) if slansky_groups is not None else ''
# seat_save_string = '_' + '_'.join([''.join([x for x in y]) for y in seat_groups]) if seat_groups is not None else ''
# stack_save_string = '_' + '_'.join([''.join([x for x in y]) for y in stack_groups]) if stack_groups is not None else ''
#
# with open('prob_slansky_123_4567_89_seat_stack.json', 'r') as f:
#     prob_dict = json.load(f)
# with open('payoff_slansky_123_4567_89_seat_stack.json', 'r') as f:
#     payoff_dict = json.load(f)


# ====================== Archive ======================
# from assumption_calc_functions import calc_exp_loss_wins, calc_prob_winning_slansky_rank


# --- Run functions to calc values on data set
# t_loss_dict, t_win_dict = calc_exp_loss_wins(games, small_blind_f=50, big_blind_f=100)
# exp_loss_seat_dict = dict()
# for t_pos, t_vals in t_loss_dict.items():
#     exp_loss_seat_dict.update({t_pos: t_vals['sum']/t_vals['count']})
# exp_loss_seat_dict['1'] = exp_loss_seat_dict['small_excl']    # use calculation excluding blinds
# exp_loss_seat_dict.pop('small_excl')
# exp_loss_seat_dict['2'] = exp_loss_seat_dict['big_excl']    # use calculation excluding blinds
# exp_loss_seat_dict.pop('big_excl')
#
# exp_win_seat_dict = dict()
# for t_pos, t_vals in t_win_dict.items():
#     exp_win_seat_dict.update({t_pos: t_vals['sum']/t_vals['count']})
# exp_win_seat_dict['2'] = exp_win_seat_dict['blinds_excl']    # use calculation excluding situation where big blind wins by default (all fold pre flop)
# exp_win_seat_dict.pop('blinds_excl')
#
# exp_loss_seat_dict = {'1': -953.5334448160535, '2': -717.590701535907, '3': -796.9296875, '4': -903.4153225806451, '5': -958.1126914660831, '6': -989.2040636042403}
# exp_win_seat_dict = {'1': 877.7987273945078, '2': 573.2828232971373, '3': 620.1804255319149, '4': 617.9281487743026, '5': 758.4563303994584, '6': 740.3076459963877}


# --- Run functions to calc values on data set
# t_slansky_prob_dict_f = calc_prob_winning_slansky_rank(games, small_blind_f=50, big_blind_f=100)
# slansky_prob_dict = dict()
# for t_rank, t_vals in t_slansky_prob_dict_f.items():
#     slansky_prob_dict.update({t_rank: t_vals['win']/t_vals['count']})
#
# --- check
# for t_key in slansky_prob_dict.keys():
#     print('Exp. value for Slansky rank %s: %f' % (t_key, (slansky_prob_dict[t_key] * np.mean(list(exp_win_seat_dict.values())) + (1 - slansky_prob_dict[t_key]) * np.mean(list(exp_loss_seat_dict.values())))))
#
# slansky_prob_dict = {'1': 0.6569037656903766, '2': 0.532046332046332, '3': 0.458252427184466, '4': 0.375, '5': 0.2822977201609298,
#                      '6': 0.21596578759800428, '7': 0.14495040577096482, '8': 0.0891238670694864, '9': 0.046566386880523}
