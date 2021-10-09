import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt


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


def create_ordered_lists(payoff_dict_f, prob_dict_f):
    ordered_lists_by_player = dict()
    util_diff_by_player = dict()
    omega_list = [x / 100 for x in range(-25, 1000, 25)]
    for p in payoff_dict_f.keys():  # [p.name for p in players_f]:
        ordered_list = list()
        non_ordered_list = list()
        utilplay_less_utilfold = list()
        for g, values in payoff_dict_f[p].items():
            for h in values.keys():
                shift_scale_play_win = (payoff_dict_f[p][g][h]['play']['win'] - payoff_dict_f[p][g][h]['play'][
                    'lose'] + 100) / 100
                shift_scale_play_lose = (payoff_dict_f[p][g][h]['play']['lose'] - payoff_dict_f[p][g][h]['play'][
                    'lose'] + 100) / 100
                shift_scale_fold = (payoff_dict_f[p][g][h]['fold']['lose'] - payoff_dict_f[p][g][h]['play'][
                    'lose'] + 100) / 100
                if not ((shift_scale_fold > shift_scale_play_lose) & (shift_scale_fold < shift_scale_play_win)):
                    print('For %s game %s hand %s not ordered' % (p, g, h))

                utilplay_less_utilfold.append(
                    (g, h, [calc_CRRA_utility([{'prob': prob_dict_f[p][g][h], 'payoff': shift_scale_play_win},
                                               {'prob': 1 - prob_dict_f[p][g][h], 'payoff': shift_scale_play_lose}],
                                              o) -
                            calc_CRRA_utility([{'prob': 1, 'payoff': shift_scale_fold}], o)
                            for o in omega_list])
                )

                if (payoff_dict_f[p][g][h]['fold']['lose'] > payoff_dict_f[p][g][h]['play']['lose']) \
                        & (payoff_dict_f[p][g][h]['fold']['lose'] < payoff_dict_f[p][g][h]['play']['win']):
                    ordered_list.append((g, h))
                else:
                    non_ordered_list.append((g, h))
            del shift_scale_play_win, shift_scale_play_lose, shift_scale_fold, h
        del g, values
        print('%s hands: ordered = %d, non-ordered = %d' % (p, len(ordered_list), len(non_ordered_list)))
        ordered_lists_by_player.update({p: {'ordered': ordered_list, 'non': non_ordered_list}})
        util_diff_by_player.update({p: utilplay_less_utilfold})
    del ordered_list, non_ordered_list, utilplay_less_utilfold

    return util_diff_by_player


def check_util_diff_monotonicity(util_diff_by_player_f):
    increase_decrease_switch = dict()
    for p in util_diff_by_player_f.keys():
        always_increasing = list()
        always_decreasing = list()
        diff_inc_dec_switch_omega_loc_index = list()
        for util_diff in util_diff_by_player_f[p]:
            inc_TF = [(util_diff[2][i] - util_diff[2][i - 1]) > 0 for i in range(1, len(util_diff[2]))]
            if all(inc_TF):
                always_increasing.append((util_diff[0], util_diff[1]))
            elif all([not x for x in inc_TF]):
                always_decreasing.append((util_diff[0], util_diff[1]))
            else:
                diff_inc_dec_switch_omega_loc_index.append(
                    [i - 1 for i in range(1, len(inc_TF)) if inc_TF[i] != inc_TF[i - 1]])
            del inc_TF
        del util_diff
        increase_decrease_switch.update(
            {p: {'always_increase': always_increasing, 'always_decrease': always_decreasing,
                 'switch': diff_inc_dec_switch_omega_loc_index}})

        if (len(always_increasing) + len(always_decreasing) + len(diff_inc_dec_switch_omega_loc_index)) != len(
                util_diff_by_player_f[p]):
            print('Warning: check designation of utility always increasing or decreasing for %s' % p)
    del always_decreasing, always_increasing, p

# payoff_win_f=tplay_win_payoff
# payoff_lose_f=tplay_lose_payoff
# payoff_fold_f=tfold_lose_payoff
# prob_win_f=tplay_win_prob
# lambda_list_f=[2]
# omega_list_f=[x / 100 for x in range(-500, 6000, 10)]
def identify_omega_range(payoff_win_f, payoff_lose_f, payoff_fold_f, prob_win_f, omega_list_f, lambda_list_f=[1.5], eps_f=1e-2, window_ass_f=5, eps_ass_f=1e-4, perc_thresh_f=0.1):
    # --- description of argument parameters ----:
    # eps_ass = 1e-3  # convergence toleration towards 50% assymptotic behavior for RELATIVE successive differences in predicted percentages
    # window_ass = 3  # number of consecutive observations meeting convergence epsilon tolerance to consider behavior assymptotic
    # eps = 1e-2 # boundary around key percentages to consider it equiavalent (100%, 0%, 50%)
    # lambda_f = 1.5    # default value to fix lambda at for check percentage convergence to bounds of 100%/0%/50%
    # perc_thresh = 0.1     # or condition to terminate assymptotic behavior...just to ensure at 50% assymptote not 0/100% (set to 50% +/- 10% by default)
    def check_convergence_to_50(x_ff, i_ff, prob_at_omega_ff, window_ass_ff=window_ass_f, eps_ff=eps_f,
                                perc_thresh_ff=perc_thresh_f, eps_ass_ff=eps_ass_f):
        return (((x_ff > (0.5 - eps_ff)) and (x_ff < (0.5 + eps_ff))) and
         (all([(abs(
             (y_ff - prob_at_omega_ff[i_ff - window_ass_ff:i_ff][j_ff - 1]) /
             prob_at_omega_ff[i_ff - window_ass_ff:i_ff][j_ff - 1]) < eps_ass_ff)
               and
               ((y_ff > (0.5 - perc_thresh_ff)) & (y_ff < (0.5 + perc_thresh_ff)))
               for j_ff, y_ff in
               enumerate(prob_at_omega_ff[i_ff - window_ass_ff:i_ff])
               ])
          and (i_ff > window_ass_ff)
          )
         )

    prob_at_lambda = dict()
    for l in lambda_list_f:
        prob_at_omega = list()
        play_util = list()
        fold_util = list()
        i_ff = 0
        for o in omega_list_f:
            t_play_util = calc_CRRA_utility([{'prob': prob_win_f, 'payoff': payoff_win_f},
                                             {'prob': 1 - prob_win_f, 'payoff': payoff_lose_f}], o)
            t_fold_util = calc_CRRA_utility([{'prob': 1, 'payoff': payoff_fold_f}], o)
            t_prob = calc_logit_prob(t_play_util, [t_play_util, t_fold_util], l)
            prob_at_omega.append(t_prob)
            play_util.append(t_play_util)
            fold_util.append(t_fold_util)
            if check_convergence_to_50(t_prob, i_ff, prob_at_omega):
                break
            i_ff += 1
        del o, t_play_util, t_fold_util, t_prob, i_ff

        # find first location prob is within epsilon of 0/100
        t_min_omega = max([i for i, x in enumerate(prob_at_omega) if ((x >= (1 - eps_f)) or (x <= (0 + eps_f)))])
        # find first location starting at the end that prob is within epsilon of 50%
        t_max_omega = min([i for i, x in enumerate(prob_at_omega) if check_convergence_to_50(x, i, prob_at_omega)])  # last condition is to ensure stability / assymptotic behavior vs. crossing 50%

        prob_at_lambda.update({l: {'min_omega': omega_list_f[t_min_omega],
                                   'min_omega_prob': prob_at_omega[t_min_omega],
                                   'max_omega': omega_list_f[t_max_omega],
                                   'max_omega_prob': prob_at_omega[t_max_omega],
                                   'prob_util': {'play_util': play_util[t_min_omega:(t_max_omega + 1)],
                                                 'fold_util': fold_util[t_min_omega:(t_max_omega + 1)], 'prob': prob_at_omega[t_min_omega:(t_max_omega + 1)]}}})
        del t_min_omega, t_max_omega
    del l

    return prob_at_lambda


def classify_gamble_types(play_util_vec_f, fold_util_vec_f):
    t_play_util_greater_than_fold = [play_util_vec_f[i] > fold_util_vec_f[i] for i in range(len(play_util_vec_f))]
    if all(t_play_util_greater_than_fold):
        # all positive util diff (riskier always preferred)
        return 'risky_dominant'
    elif all([not x for x in t_play_util_greater_than_fold]):
        # all negative util diff (safer always preferred)
        return 'safe_dominant'
    elif (not all(t_play_util_greater_than_fold)) and (
            not all([not x for x in t_play_util_greater_than_fold])) and (not t_play_util_greater_than_fold[0]) and t_play_util_greater_than_fold[-1]:
        # mix:  y > x -----> x > y in increasing omega (shouldn't happen)
        print('CHECK: in game %s hand %s there is increasing probability of choosing play (riskier) with increasing levels of risk aversion (omega)')
        return 'prob_risk_increases_omega_increases'
    elif (not all(t_play_util_greater_than_fold)) and (not all([not x for x in t_play_util_greater_than_fold])) and t_play_util_greater_than_fold[0] and (not t_play_util_greater_than_fold[-1]):
        # mix: x > y ------> y> x in increasing omega
        return 'prob_risk_decreases_omega_increases'
    else:
        print('WARNING: uncategorized case')
        return 'ERROR'


def main():
    def plot_histograms_of_probs(omega_ranges_f, select_lambdas_f=[1.5]):
        for l in select_lambdas_f:
            min_omegas = list()
            min_prob = list()
            max_omegas = list()
            max_prob = list()
            for p in omega_ranges_f.keys():
                for g in omega_ranges_f[p].keys():
                    for h in omega_ranges_f[p][g].keys():
                        min_omegas.append(omega_ranges_f[p][g][h][l]['min_omega'])
                        min_prob.append(omega_ranges_f[p][g][h][l]['min_omega_prob'])
                        max_omegas.append(omega_ranges_f[p][g][h][l]['max_omega'])
                        max_prob.append(omega_ranges_f[p][g][h][l]['max_omega_prob'])
            for k, v in {'lb estimated omegas': min_omegas, 'lb of estimated probs': min_prob, 'ub estimated omegas': max_omegas, 'ub of estimated probs': max_prob}.items():
                plt.figure()
                plt.hist(v)
                plt.title('%s at lambda = %3.2f' % (str(k), l))

    # ----- File I/O params -----
    fp_output = 'output'
    fn_prob_dict = 'prob_dict_dnn.json'
    fn_payoff_dict = 'payoff_dict_dnn.json'

    # ----- LOAD DATA -----
    # game data
    with open("python_hand_data.pickle", 'rb') as f:
        data = pickle.load(f)
    players = data['players']
    
    with open(os.path.join(fp_output, fn_prob_dict), 'r') as f:
        prob_dict = json.load(f)
    with open(os.path.join(fp_output, fn_payoff_dict), 'r') as f:
        payoff_dict = json.load(f)

    # --- check to see if all gambles in data set are ordered
    # util_diff_by_player = create_ordered_lists(payoff_dict, prob_dict)
    
    # check if difference in preferences anywhere in the tested domain (x > y ----> y > x)
    # check_util_diff_monotonicity(util_diff_by_player)

    # holding lambda fixed, find bounds of omega for shares go to 0/100% and converge to 50%
    # ---- for range of omegas, calc probabilities
    omega_list = [x / 100 for x in range(-500, 2000, 20)]
    payoff_shift = 100
    payoff_scale = 100
    lambda_list = [0.1, 0.5, 1, 1.5, 2]

    omega_ranges = {p: {g: {h: {} for h in values.keys()} for g, values in payoff_dict[p].items()} for p in
                    [p.name for p in players]}
    for p in omega_ranges.keys():
        print('Calculating range for player %s' % p)
        for g, values in payoff_dict[p].items():
            for h in values.keys():
                shift_scale_play_win = (payoff_dict[p][g][h]['play']['win'] - payoff_dict[p][g][h]['play'][
                    'lose'] + payoff_shift) / payoff_scale
                shift_scale_play_lose = (payoff_dict[p][g][h]['play']['lose'] - payoff_dict[p][g][h]['play'][
                    'lose'] + payoff_shift) / payoff_scale
                shift_scale_fold = (payoff_dict[p][g][h]['fold']['lose'] - payoff_dict[p][g][h]['play'][
                    'lose'] + payoff_shift) / payoff_scale

                omega_ranges[p][g][h].update(
                    identify_omega_range(shift_scale_play_win, shift_scale_play_lose, shift_scale_fold, prob_dict[p][g][h], omega_list, lambda_list))

                # del shift_scale_play_win, shift_scale_play_lose, shift_scale_fold
            del h
        del g, values
    del p

    # try:
    #     plot_histograms_of_probs(omega_ranges, select_lambdas_f=[1.5])  # can use any lambdas used in lambda loop. default is 1.5 but this must have been used in loop
    # except KeyError:
    #     print('Tried to plot histograms but likely invalid lambda level specified')

    # lambda_ranges = 

    fold_dominates = {p: {l: list() for l in list(set([l for player in omega_ranges.values() for game in player.values() for hand in game.values() for l in hand.keys()]))} for p in omega_ranges.keys()}
    play_dominates = {p: {l: list() for l in list(set([l for player in omega_ranges.values() for game in player.values() for hand in game.values() for l in hand.keys()]))} for p in omega_ranges.keys()}
    prob_play_increases_in_risk_aversion = {p: {l: list() for l in list(set([l for player in omega_ranges.values() for game in player.values() for hand in game.values() for l in hand.keys()]))} for p in omega_ranges.keys()}
    prob_play_decreases_in_risk_aversion = {p: {l: list() for l in list(set([l for player in omega_ranges.values() for game in player.values() for hand in game.values() for l in hand.keys()]))} for p in omega_ranges.keys()}
    for p in omega_ranges.keys():
        print('\n\n=========== %s =============' % p)
        for l in list(sorted(set([l for player in omega_ranges.values() for game in player.values() for hand in game.values() for l in hand.keys()]))):
            t_hand_count = 0
            for g in omega_ranges[p].keys():
                for h in omega_ranges[p][g].keys():
                    t_hand_count += 1
                    # t_play_util_greater_than_fold = [omega_ranges[p][g][h]['prob_util']['play_util'][i] >
                    #                                  omega_ranges[p][g][h]['prob_util']['fold_util'][i] for i in
                    #                                  range(len(omega_ranges[p][g][h]['prob_util']['play_util']))]

                    t_class = classify_gamble_types(omega_ranges[p][g][h][l]['prob_util']['play_util'], omega_ranges[p][g][h][l]['prob_util']['fold_util'])
                    # omega_ranges[p][g][h]['prob_util']
                    if t_class == 'risky_dominant':
                        play_dominates[p][l].append((g, h))
                    elif t_class == 'safe_dominant':
                        fold_dominates[p][l].append((g, h))
                    elif t_class == 'prob_risk_increases_omega_increases':
                        prob_play_increases_in_risk_aversion[p][l].append((g, h))
                    elif t_class == 'prob_risk_decreases_omega_increases':
                        prob_play_decreases_in_risk_aversion[p][l].append((g, h))
                    else:
                        print('WARNING: uncategorized case for player %p game %s hand %s lambda level=%3.2f' % (p, g, h, l))
            del h
        
            if (len(play_dominates[p][l]) + len(fold_dominates[p][l]) + len(prob_play_increases_in_risk_aversion[p][l]) + len(
                    prob_play_decreases_in_risk_aversion[p][l])) != t_hand_count:
                print('WARNING: check categorization for completeness and no overlap')

            del g, t_hand_count

            print('-----Lambda level %3.2f-----' % l)
            print(
                '-- %s hand categorization count (%d total)\n\tplay util dominates: %d\n\tfold util dominates: %d\n\tmixed, play for low risk aversion: %d\n\tmixed: play for high omega (SHOULD BE ZERO): %d' % (
                p, len(play_dominates[p][l]) + len(fold_dominates[p][l]) + len(prob_play_increases_in_risk_aversion[p][l]) + len(
                    prob_play_decreases_in_risk_aversion[p][l]), len(play_dominates[p][l]), len(fold_dominates[p][l]),
                len(prob_play_decreases_in_risk_aversion[p][l]), len(prob_play_increases_in_risk_aversion[p][l])))
        del l
    del p

    return None


# main()
