import json
import copy
import numpy as np
import pandas as pd
import random
import os
import pickle

import matplotlib.pyplot as plt

from RUM_functions_classes import RandomUtilityModel, ChoiceSituation, Option
from RUM_functions_classes import generate_choice_situations, generate_synthetic_data, reformat_choice_situations_for_model, calc_CRRA_utility, run_multistart, parse_multistart, calc_robust_varcov, calc_mle_tstat, check_likelihood, calc_RUM_prob
from assumption_calc_functions import two_sample_test_prop, two_sample_test_ind_means
from control_params import control_param_dict, ControlParams, results_param_dict, ResultsParams

pd.options.display.max_columns = 25


def load_base_data(fpn_game_data, fpn_prob_dict, fpn_payoff_dict):
    # data that is not model dependent
    # ----- LOAD DATA -----
    # game data
    with open(fpn_game_data, 'rb') as f:
        data = pickle.load(f)
    players = data['players']
    games = data['games']

    try:
        with open(fpn_prob_dict, 'r') as f:
            prob_dict = json.load(f)
        with open(fpn_payoff_dict, 'r') as f:
            payoff_dict = json.load(f)
    except FileNotFoundError:
        print('No probability and payoff dictionaries saved down in output folder')
    return players, games, prob_dict, payoff_dict


def load_choice_situation_data(players, prob_dict, payoff_dict, choice_situations_dir_save_string, select_player_list_save_string, select_player_list, save_TF=False):
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
            # game_hand_player_index = create_game_hand_index(players[select_player_index])

            # ---- actual data ----
            choice_situations.append(generate_choice_situations(player_f=players[select_player_index], payoff_dict_f=payoff_dict, prob_dict_f=prob_dict))
        del select_player
        choice_situations = [cs for p in choice_situations for cs in p]
        # add gamble type evaluations
        for cs in choice_situations:
            cs.CRRA_ordered_gamble_info = cs.get_gamble_info(lambda_list_f=[0.5, 10])
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

    # --- print summary in info ---
    print('Count of choice situation types:')
    t_counts = {'post_win': 0, 'post_loss': 0, 'post_neutral': 0, 'post_loss_excl_blind_only': 0,
                'post_win_excl_blind_only': 0, 'post_neutral_or_blind_only': 0}
    unclassified = 0
    for cs in choice_situations:
        for k in t_counts.keys():
            if cs.__getattribute__(k) is not None:
                t_counts[k] += int(cs.__getattribute__(k))
        unclassified += all([cs.__getattribute__(k) is None for k in t_counts.keys()])
    for k, v in t_counts.items():
        print('\t%s: %s' % (k, v))
    print('\tUnclassified: %d' % unclassified)

    return choice_situations


def load_multistart(multi_start_dir_save_string):
    try:
        # load from saved
        select_results = list()
        select_models = list()
        for fn in [f for f in os.listdir(multi_start_dir_save_string) if os.path.isfile(os.path.join(multi_start_dir_save_string, f)) if f[0] != '.']:  # os.listdir(save_path):
            with open(os.path.join(multi_start_dir_save_string, fn), 'rb') as f:
                t_input = pickle.load(f)
                select_results = select_results + t_input['results']
                select_models.append(t_input['model'])
        return select_results, select_models
    except FileNotFoundError as e:
        print('No available results with specified configuration\n%s' % e)


def inspect_gambles_by_type(choice_situations, select_cases, case_comparison, select_plot_cases, plot_TF=False):
    # --- collect data
    choice_situations_by_case = {'all': choice_situations}  #, 'post loss': copy.deepcopy([cs for cs in choice_situations if cs.__getattribute__('post_loss_excl_blind_only')]), 'post neutral': copy.deepcopy([cs for cs in choice_situations if cs.__getattribute__('post_neutral_or_blind_only')])}
    choice_situations_by_case.update({case: copy.deepcopy([cs for cs in choice_situations if cs.__getattribute__(case)]) for case in select_cases})
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
    for case in case_comparison:
        print('\n\n==== COMPARING CASES %s and %s =====' % (case[0], case[1]))
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
    if plot_TF:
        for case in select_plot_cases:
            t_select_choice_situations = choice_situations_by_case[case]
            for l in gamble_type_summary[case].keys():
                for t_type in gamble_type_summary[case][l].keys():
                    plt.figure()
                    plt.title('%s at lambda=%3.2f' % (t_type, l))
                    for t_ind in gamble_type_summary[case][l][t_type]:
                        if t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['classification_warning']:
                            print(
                            'WARNING: game %s hand %s at lambda=%3.2f classified as %s but some conditions are not met for this classification' % (
                                t_select_choice_situations[t_ind].tags['game'],
                                t_select_choice_situations[t_ind].tags['hand'],
                                l,
                                t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l]['gamble_type']))
                        try:
                            t_select_choice_situations[t_ind].plot_single_prob_line(t_select_choice_situations[t_ind].CRRA_ordered_gamble_info[l])
                        except ZeroDivisionError:
                            print('check calc of t_omega range for lambda = %s gamble_type = %s' % (l, t_type))
                    del t_ind
                del t_type
            del l
        del case


def subselect_data(select_case, choice_situations, select_gamble_types, fraction_of_data_to_use_for_estimation, reformat_print_obs_summary=True):
    # ---- preprocess: partition data and sub-sample -----
    if select_case != 'all':
        t_candidates = [cs for cs in choice_situations if (cs.__getattribute__(select_case) and (
                cs.CRRA_ordered_gamble_type in select_gamble_types))]  # any([v['gamble_type'] in select_gamble_types for l, v in cs.CRRA_ordered_gamble_type.items()]))]
    else:
        t_candidates = [cs for cs in choice_situations if (
                cs.CRRA_ordered_gamble_type in select_gamble_types)]  # any([v['gamble_type'] in select_gamble_types for l, v in cs.CRRA_ordered_gamble_type.items()])]
    choice_param_dictionary = reformat_choice_situations_for_model(
        random.sample(t_candidates, round(fraction_of_data_to_use_for_estimation * len(t_candidates))), print_obs_summary=reformat_print_obs_summary)
    omega_max_95percentile = np.quantile(
        [cs.CRRA_ordered_gamble_info[l]['max_omega'] for l in t_candidates[0].CRRA_ordered_gamble_info.keys() for cs in
         t_candidates], 0.95)
    del t_candidates

    return choice_param_dictionary, omega_max_95percentile


def filter_multistarts(list_dict_params, list_dict_obs, opt_param_names, lb, ub, lb_additional=None, ub_additional=None, eps=1e-2, tstat_limit=1.96, plot_TF=False):
    def filter(df_params, opt_param_names, lb, ub, lb_additional, ub_additional, eps, tstat_limit):
        t_filter = pd.Series(True, index=df_params.index)
        for n in opt_param_names:  #
            # print(n)
            filter_bounds = (df_params[n] >= (lb[n] + eps)) & (df_params[n] <= (ub[n] - eps))
            filter_bounds_additional = (df_params[n] >= (lb_additional[n] + eps)) & (
                    df_params[n] <= (ub_additional[n] - eps))
            filter_tstat = (df_params[n + '_tstat'].abs() >= tstat_limit)
            filter_initial = (df_params[n] >= (df_params[n + '_initial'] + eps)) | (
                    df_params[n] <= (df_params[n + '_initial'] - eps))
            t_filter = t_filter & filter_bounds & filter_bounds_additional & filter_initial & filter_tstat
            print('After %s, %d obs remaining' % (n, sum(t_filter)))
        if 'lambda' in opt_param_names:
            filter_lambda1 = (df_params['lambda'] > (1 + eps)) | (df_params['lambda'] < (1 - eps))
        else:
            filter_lambda1 = pd.Series(True, index=df_params.index)
        if 'omega' in opt_param_names:
            filter_omega1 = (df_params['omega'] > (1 + eps)) | (df_params['omega'] < (1 - eps))

        filter_message = ((
                df_params.message == b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'))   # | (df_params.message == b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'))
        ind_filter = df_params.index[t_filter & filter_omega1 & filter_lambda1 & filter_message]  #
        print('Across all filters %d/%d observations remain' % (len(ind_filter), df_params.shape[0]))
        return ind_filter

    if lb_additional is None:
        lb_additional = lb
    if ub_additional is None:
        ub_additional = ub

    # create data frames of estimated multistart parameters
    df_params = pd.DataFrame(list_dict_params).set_index('est_run_id')
    df_obs = pd.DataFrame(list_dict_obs)

    # apply filters for valid convergence
    ind_filter = filter(df_params, opt_param_names, lb, ub, lb_additional, ub_additional, eps, tstat_limit)
    df_obs_filtered = df_obs.loc[df_obs.est_run_id.isin(list(ind_filter))]

    # Summarize estimates
    print('---- AFTER FILTERING FOR VALID CONVERGENCE-------')
    print('n obs = %d' % len(ind_filter))
    for p in opt_param_names:
        print('mean %s = %3.2f' % (p.upper(), df_params.loc[ind_filter][p].mean()))
        print('stdev %s = %3.5f' % (p.upper(), df_params.loc[ind_filter][p].std()))

    if plot_TF:
        # Histogram of estimated parameters
        plt.figure()
        plt.hist([x['omega'] for x in list_dict_params])
        plt.title('omega before filtering')
        plt.show()

        for n in opt_param_names:
            plt.figure()
            df_params.loc[ind_filter][n].hist()
            plt.title('Histogram of ' + n + ' estimates')
            plt.show()

        # --- Predictions using estimations
        # play v fold utilities
        plt.figure()
        plt.scatter(df_obs_filtered.loc[df_obs_filtered.actual_share == 0, 'util_fold'],
                    df_obs_filtered.loc[df_obs_filtered.actual_share == 0, 'util_play'], c='r')
        plt.scatter(df_obs_filtered.loc[df_obs_filtered.actual_share == 1, 'util_fold'],
                    df_obs_filtered.loc[df_obs_filtered.actual_share == 1, 'util_play'], c='b')
        plt.legend(['actually folded', 'actually played'])
        plt.plot([min(min(df_obs_filtered['util_fold']), min(df_obs_filtered['util_play'])),
                  max(max(df_obs_filtered['util_fold']), max(df_obs_filtered['util_play']))],
                 [min(min(df_obs_filtered['util_fold']), min(df_obs_filtered['util_play'])),
                  max(max(df_obs_filtered['util_fold']), max(df_obs_filtered['util_play']))], '.-')
        plt.xlabel('Utility of folding at estimated parameters')
        plt.ylabel('Utility of playing at estimated parameters')
        plt.show()

        # AVP
        plt.figure()
        plt.scatter(df_obs_filtered['pred_share'], df_obs_filtered['actual_share'])
        plt.xlabel('Predicted share')
        plt.ylabel('Choice')
        plt.title('Actual v. predicted shares')
        plt.show()

    return df_params.loc[ind_filter], df_obs_filtered


def calc_util_rational(exp_omega, choice_param_dictionary):
    for rank in choice_param_dictionary.keys():
        for seat in choice_param_dictionary[rank].keys():
            for item in choice_param_dictionary[rank][seat]:
                t_play = calc_CRRA_utility(item['params']['play'], exp_omega)
                t_fold = calc_CRRA_utility(item['params']['fold'], exp_omega)
                t_rational_TF = ((t_play > t_fold) and bool(item['n_chosen']['play'])) or (
                        (t_fold > t_play) and bool(item['n_chosen']['fold']))
                if bool(item['n_chosen']['play']) and t_rational_TF:
                    t_type = 'rational_play'
                elif bool(item['n_chosen']['play']) and not t_rational_TF:
                    t_type = 'irrational_play'
                elif bool(item['n_chosen']['fold']) and t_rational_TF:
                    t_type = 'rational_fold'
                elif bool(item['n_chosen']['fold']) and not t_rational_TF:
                    t_type = 'irrational_fold'
                else:
                    print('Error in categorizing choices')
                item.update({'exp_util_omega_' + str(int(round(exp_omega, 4) * 10000)) + 'e4': {
                    'omega': exp_omega,
                    'play': t_play,
                    'fold': t_fold,
                    'rational_TF': t_rational_TF,
                    'rational_choice_type': t_type}})
    return choice_param_dictionary, 'exp_util_omega_' + str(int(round(exp_omega, 4) * 10000)) + 'e4'


def count_rational_action_types(choice_param_dictionary, exp_omega_key, print_summary_TF=True):
    t_list = list()
    for rank in choice_param_dictionary.keys():
        for seat in choice_param_dictionary[rank].keys():
            for item in choice_param_dictionary[rank][seat]:
                t_list.append(item[exp_omega_key]['rational_choice_type'])
    t_dict = {k: sum([x == k for x in t_list]) for k in set(t_list)}
    # --- calculate utilities and expected value of the parameters

    total_play = t_dict['rational_play'] + t_dict['irrational_play']
    total_fold = t_dict['rational_fold'] + t_dict['irrational_fold']
    total_rational = t_dict['rational_play'] + t_dict['rational_fold']
    total_irrational = t_dict['irrational_play'] + t_dict['irrational_fold']
    total_should_have_played = t_dict['rational_play'] + t_dict['irrational_fold']
    total_should_have_folded = t_dict['rational_fold'] + t_dict['irrational_play']
    total_obs = sum(t_dict.values())
    if ((total_rational + total_irrational) != total_obs) or ((total_play + total_fold) != total_obs) or ((total_should_have_played + total_should_have_folded != total_obs)):
        print('WARNING: CHECK RATIONAL ACTION COUNTS')

    if print_summary_TF:
        print('--- Rational choice:')
        print('(Rational) Blended: %3.1f%% (n=%d):' % (
            total_rational / total_obs * 100, total_obs))
        print('(Rational)   Play - positive expected value: %3.1f%% (n=%d)' % (
            t_dict['rational_play'] / total_should_have_played * 100, t_dict['rational_play']))
        print('(Irrational) Fold - positive expected value: %3.1f%% (n=%d)' % (
            t_dict['irrational_fold'] / total_should_have_played * 100, t_dict['irrational_fold']))
        print('(Irrational)   Play - negative expected value: %3.1f%% (n=%d)' % (
            t_dict['irrational_play'] / total_should_have_folded * 100, t_dict['irrational_play']))
        print('(Rational) Fold - negative expected value: %3.1f%% (n=%d)' % (
            t_dict['rational_fold'] / total_should_have_folded * 100, t_dict['rational_fold']))

    return t_dict


def save_parameter_estimates(df_params, rationality_summary, select_model_param_names, param_estimates_dir_save_string, select_player_list_save_string, select_case, save_TF=True):
    # save estimated parameter output
    if save_TF:
        try:
            with open(os.path.join(param_estimates_dir_save_string,
                                   select_player_list_save_string + '_' + select_case + '.json'), 'w') as f:
                t_dict = {p: {'mean': df_params[p].mean(), 'stdev': df_params[p].std(),
                              'nobs': float(df_params[p].count())} for p in select_model_param_names}
                t_dict.update(rationality_summary)
                json.dump(t_dict, fp=f)
                del t_dict
            df_params.to_csv(os.path.join(param_estimates_dir_save_string,
                                          select_player_list_save_string + '_' + select_case + '.csv'))
        except FileNotFoundError:
            try:
                os.makedirs(param_estimates_dir_save_string)
                with open(os.path.join(param_estimates_dir_save_string,
                                       select_player_list_save_string + '_' + select_case + '.json'), 'w') as f:
                    t_dict = {p: {'mean': df_params[p].mean(), 'stdev': df_params[p].std(),
                                  'nobs': float(df_params[p].count())} for p in select_model_param_names}
                    t_dict.update(rationality_summary)
                    json.dump(t_dict, fp=f)
                    del t_dict
                df_params.to_csv(os.path.join(param_estimates_dir_save_string,
                                              select_player_list_save_string + '_' + select_case + '.csv'))
            except FileExistsError as e:
                print(e)


def aggregate_param_estimates(rp, save_TF=False):
    list_df_params_select = list()
    list_df_params_all = list()
    for player in rp.select_player_list:
        if not isinstance(player, list):
            rp.set_player_list_save_string([player])
        else:
            rp.set_player_list_save_string(player)
        rp.set_param_estimates_dir_save_string(rp.select_player_list_save_string)
        for state in rp.select_states:
            # load in estimates
            try:
                t_select_case = state + '_select' + str(rp.num_param_estimates)
                t_df_select = pd.read_csv(os.path.join(rp.param_estimates_dir_save_string,
                                                rp.select_player_list_save_string + '_' + t_select_case + '.csv'))
                t_df_select['player'] = pd.Series([player] * t_df_select.shape[0])
                t_df_select['state'] = pd.Series([state] * t_df_select.shape[0])
                list_df_params_select.append(t_df_select)

                t_df_select_all = pd.read_csv(os.path.join(rp.param_estimates_dir_save_string,
                                                     rp.select_player_list_save_string + '_' + state + '.csv'))
                t_df_select_all['player'] = pd.Series([player] * t_df_select_all.shape[0])
                t_df_select_all['state'] = pd.Series([state] * t_df_select_all.shape[0])
                list_df_params_all.append(t_df_select_all)

                del t_select_case, t_df_select, t_df_select_all
            except FileNotFoundError as e:
                print('%s for player %s state %s' % (e, player, state))

    df_agg_params = pd.DataFrame()
    for t_df_select in list_df_params_select:
        df_agg_params = pd.concat([df_agg_params, t_df_select], ignore_index=True)
    df_agg_params.drop(columns=['Unnamed: 0'], inplace=True)

    df_agg_params_all = pd.DataFrame()
    for t_df_select_all in list_df_params_all:
        df_agg_params_all = pd.concat([df_agg_params_all, t_df_select_all])
    # df_agg_params_all.drop(columns=['Unnamed: 0'], inplace=True)

    if save_TF:
        try:
            df_agg_params.to_csv(os.path.join(rp.results_save_string, 'param_estimates_select.csv'))
            df_agg_params_all.to_csv(os.path.join(rp.results_save_string, 'param_estimates_all.csv'))
        except FileNotFoundError:
            try:
                os.makedirs(os.path.join(rp.results_save_string))
                df_agg_params.to_csv(os.path.join(rp.results_save_string, 'param_estimates_select.csv'))
                df_agg_params_all.to_csv(os.path.join(rp.results_save_string, 'param_estimates_all.csv'))
            except FileExistsError as e:
                print(e)
    return df_agg_params, df_agg_params_all


def main(cp):
    # (OPT) ===== SET CONTROL PARAMS =====
    print('Value of control params: %s\n' % cp.print_params())

    # ====== IMPORT DATA ======
    # (REQ) ---- base game data (non-model dependent)
    players, games, prob_dict, payoff_dict = load_base_data("python_hand_data.pickle",
                                                            os.path.join(cp.fp_output, cp.fn_prob_dict),
                                                            os.path.join(cp.fp_output, cp.fn_payoff_dict))

    # (REQ) ---- choice situation data (if not available in memory, then create
    choice_situations = load_choice_situation_data(players, prob_dict, payoff_dict, cp.choice_situations_dir_save_string, cp.select_player_list_save_string, cp.select_player_list, save_TF=cp.save_TF)

    # (OPT) --- inspect choice situation gamble type info (OPTIONAL, can turn off)
    # inspect_gambles_by_type(choice_situations,
    #                         select_cases=['post_neutral_or_blind_only', 'post_loss_excl_blind_only', 'post_win_excl_blind_only'],
    #                         case_comparison=[('post_neutral_or_blind_only', 'post_loss_excl_blind_only'), ('post_neutral_or_blind_only', 'post_win_excl_blind_only')],
    #                         select_plot_cases=['post_neutral_or_blind_only'], plot_TF=True)

    # (REQ) --- subselect data for certain types of gambles, previous outcomes, test set
    choice_param_dictionary, omega_max_95percentile = subselect_data(cp.select_case, choice_situations, cp.select_gamble_types, cp.fraction_of_data_to_use_for_estimation)

    # -- (OPTIONAL - SWITCH TO SYNTHETIC DATA FOR TESTING MODEL CODE, comment this out if not debugging)
    # choice_param_dictionary = generate_synthetic_data()

    # (OPT) ==== OPTIONAL INVESTIGATION =======
    model = RandomUtilityModel(data_f=choice_param_dictionary, param_names_f=cp.select_model_param_names, LL_form_f=cp.select_prob_model_type)
    model.fit(est_param_dict={'options': {'ftol': cp.ftol, 'gtol': cp.gtol, 'maxiter': cp.maxiter, 'iprint': 1}})
    print('======= Results from single instance of model estimation\n %s' % model.results)

    # check_likelihood(model, model.lb, model.ub, prob_type_f=select_prob_model_type)

    # ==== MULTISTART MODEL FITTING ==========
    # (OPT- IF ALREADY SAVED CAN JUST LOAD) --------
    # select_results = run_multistart(nstart_points=cp.num_multistarts, est_param_dictionary={'options': {'ftol': cp.ftol, 'gtol': cp.gtol, 'maxiter': cp.maxiter, 'disp': None}, 'lb': cp.lb_model, 'ub': cp.ub_model}, model_object=model, save_iter=cp.save_iter, save_path=cp.multi_start_dir_save_string, save_index_start=cp.save_index_start)

    # (OPT- IF RUNNING MULTISTART NO NEED) -----
    select_results, select_models = load_multistart(cp.multi_start_dir_save_string)

    # (REQ) ==== PARSE MULTISTARTS =======
    list_dict_params, list_dict_obs = parse_multistart(select_results, param_locs_names_f={k: select_models[0].param_names_f.index(k) for k in select_models[0].param_names_f}, choice_param_dictionary_f=choice_param_dictionary)

    df_params, df_obs = filter_multistarts(list_dict_params, list_dict_obs, select_models[0].param_names_f, cp.lb_model, cp.ub_model, lb_additional=None, ub_additional=cp.ub_additional, plot_TF=True)

    # (OPT) ==== EVALUATE RATIONALITY =====
    choice_param_dictionary, exp_omega_key = calc_util_rational(df_params['omega'].mean(), choice_param_dictionary)
    rationality_summary = count_rational_action_types(choice_param_dictionary, exp_omega_key)

    # (OPT) ==== SAVE PARAM ESTIMATES ====
    save_parameter_estimates(df_params, rationality_summary, cp.select_model_param_names, cp.param_estimates_dir_save_string, cp.select_player_list_save_string, cp.select_case, save_TF=cp.save_TF)


# main(cp=ControlParams(control_param_dict))


def create_output(rp):
    # load choice situations

    t_obs_list = list()
    t_est_list = list()
    for player in rp.select_player_list:
        if not isinstance(player, list):
            rp.set_player_list_save_string([player])
        else:
            rp.set_player_list_save_string(player)
        rp.set_param_estimates_dir_save_string(rp.select_player_list_save_string)

        choice_situations = load_choice_situation_data(None, None, None,
                                                       rp.choice_situations_dir_save_string,
                                                       rp.select_player_list_save_string, player,
                                                       save_TF=False)

        for state in rp.select_states:
            # for gamble in select_gamble_types[0]:
            choice_param_dictionary, _ = subselect_data(select_case=state, choice_situations=choice_situations,
                                                        select_gamble_types=rp.select_gamble_types,
                                                        fraction_of_data_to_use_for_estimation=1, reformat_print_obs_summary=False)

            # load in estimates
            try:
                df_params = pd.read_csv(os.path.join(rp.param_estimates_dir_save_string,
                                                     rp.select_player_list_save_string + '_' + state + '.csv'))
                df_params['select_for_output'] = pd.Series([True] * min(df_params.shape[0], rp.num_param_estimates) + [False] * (df_params.shape[0] - min(df_params.shape[0], rp.num_param_estimates)))

                #######################
                # Check number of gambles that have indifference omega less than exected value of omega
                count_mixed = 0
                count_less_than_mean = 0
                for cs in choice_situations:
                    if cs.CRRA_ordered_gamble_type == 'prob_risk_decreases_omega_increases' and cs.__getattribute__(state):
                        count_mixed += 1
                        if cs.omega_indifference is not None:
                            if cs.omega_indifference < df_params.loc[df_params.select_for_output, 'omega'].mean():
                                count_less_than_mean += 1
                print('For state %s player %s\n%d of %d (%3.1f%%) mixed gambles choice situations have indifference omega < average of selected estimates' % (
                    state, player, count_less_than_mean, count_mixed, count_less_than_mean / count_mixed * 100
                ))
                #######################

                df_select_params = df_params.loc[df_params['select_for_output']].reindex()

                choice_param_dictionary, exp_omega_key = calc_util_rational(df_select_params['omega'].mean(), choice_param_dictionary)

                save_parameter_estimates(df_select_params, rationality_summary=count_rational_action_types(choice_param_dictionary, exp_omega_key, print_summary_TF=False), select_model_param_names=rp.select_model_param_names,
                                         param_estimates_dir_save_string=rp.param_estimates_dir_save_string,
                                         select_player_list_save_string=rp.select_player_list_save_string, select_case=state + '_select' + str(df_select_params.shape[0]),
                                         save_TF=rp.save_TF)
                t_est_list.append({'player': player, 'state': state,
                                   'est': {pe: {'mean': df_select_params[pe].mean(), 'stdev': df_select_params[pe].std(), 'nobs': df_select_params.shape[0]} for pe in rp.select_model_param_names}})

                for rank in choice_param_dictionary.keys():
                    for seat in choice_param_dictionary[rank].keys():
                        temp_choice_situations = [cs for cs in choice_situations if (cs.slansky_strength == rank) and (cs.seat == seat)]
                        for item in choice_param_dictionary[rank][seat]:
                            # item = choice_param_dictionary[1][1][0] #######
                            if item['CRRA_gamble_type'] == 'prob_risk_decreases_omega_increases':
                                t_check = item['omega_indifference'] < item[exp_omega_key]['omega']
                            else:
                                t_check = None
                            t_dict = {'CRRA_gamble_type': item['CRRA_gamble_type'], 'player': player, 'seat': seat,
                                      'rank': rank, 'cs_id': item['id'], 'omega_indifference_less_than_exp': t_check, 'omega_indifference': item['omega_indifference'], 'est_omega': item[exp_omega_key]['omega']
                                      }
                            for k, outcomes in item['params'].items():
                                for i in range(len(outcomes)):
                                    t_dict.update({k + '_' + j + str(i): v for j, v in outcomes[i].items()})
                            t_dict.update(item['n_chosen'])
                            for k in ['rational_TF', 'rational_choice_type']:
                                t_dict.update({k: item[exp_omega_key][k]})
                            for k in ['play', 'fold']:
                                t_dict.update({k+'_util': item[exp_omega_key][k]})
                            t_dict.update({'prob_play': calc_RUM_prob(util_i=item[exp_omega_key]['play'], util_j=[item[exp_omega_key]['play'], item[exp_omega_key]['fold']], lambda_f=df_select_params['lambda'].mean(), kappa_f=None)})
                            t_dict.update({'state': [x for x in rp.select_states if [cs for cs in temp_choice_situations if (cs.tags['id'] == item['id'])][0].__getattribute__(x)][0]})
                            t_obs_list.append(t_dict)
                            del t_dict, t_check
                        del item, temp_choice_situations
                    del seat
                del rank

            except FileNotFoundError as e:
                print('WARNING: %s' % e)

            del choice_param_dictionary
        del state
    del player

    # create OBS data frame for export
    df_choices = pd.DataFrame(t_obs_list)
    df_choices['choice'] = 'fold'
    df_choices.loc[df_choices['play'] == 1, 'choice'] = 'play'
    df_choices.loc[df_choices['fold'] == 1, 'choice'] = 'fold'
    if any(df_choices.loc[df_choices['play'] == 1, 'choice'] != 'play') or any(
            df_choices.loc[df_choices['fold'] == 1, 'choice'] != 'fold'):
        print('WARNING: mismapping of play/fold binary columns')

    # ------------- HYPOTHESIS TESTING ------------------------
    # create estimates data frame with two sample test for difference of means for export
    t_list_comp = list()
    for i, p1 in enumerate(t_est_list):
        for j, p2 in enumerate(t_est_list):
            if j > i:
                for pe in rp.select_model_param_names:
                    t_dict = {'player1': p1['player'], 'player2': p2['player'], 'p1_state': p1['state'],
                              'p2_state': p2['state'],
                              'param': pe, 'p1_mean': p1['est'][pe]['mean'], 'p1_stdev': p1['est'][pe]['stdev'], 'p1_nobs': p1['est'][pe]['nobs'],
                              'p2_mean': p2['est'][pe]['mean'], 'p2_stdev': p2['est'][pe]['stdev'], 'p2_nobs': p2['est'][pe]['nobs']}
                    t_dict.update(dict(zip(['tstat', 'pval'], two_sample_test_ind_means(t_dict['p1_mean'],
                                               t_dict['p2_mean'],
                                               t_dict['p1_stdev'],
                                               t_dict['p2_stdev'],
                                               t_dict['p1_nobs'],
                                               t_dict['p2_nobs'], n_sides=rp.hypothesis_test_nsides, print_f=False))))
                    t_list_comp.append(t_dict)
                    del t_dict
                del pe
        del j
    del i

    # create rational hypothesis testing dataframe for export
    t_master = df_choices.reindex()
    t_master['player'] = t_master.player.apply(lambda x: '_'.join(x) if isinstance(x, list) else x)
    t_col = 'rational_TF'
    t_df = t_master.groupby(['player', 'state', 'CRRA_gamble_type']).agg({t_col: ['sum', 'count']}).droplevel(0, axis=1).reset_index()
    t_df['prop_' + t_col] = t_df['sum'] / t_df['count']

    t_list_prop = list()
    for _, row1 in t_df.iterrows():
        for _, row2 in t_df.iterrows():
            t_dict = {'player1': row1['player'], 'player2': row2['player'], 'p1_state': row1['state'], 'p2_state': row2['state'], 'p1_gamble': row1['CRRA_gamble_type'], 'p2_gamble': row2['CRRA_gamble_type']}
            t_dict.update(dict(zip(['tstat', 'pval'], two_sample_test_prop(row1['prop_' + t_col], row2['prop_' + t_col], row1['count'], row2['count'], n_sides_f=2))))
            t_list_prop.append(t_dict)
            del t_dict
        del row2
    del row1, t_df

    # create aggregated file of parameter estimates used to generate other data
    _ = aggregate_param_estimates(rp, save_TF=rp.save_TF)

    if rp.save_TF:
        try:
            with open(os.path.join(rp.results_save_string, 'choice_analysis.csv'), 'w') as f:
                df_choices.to_csv(f)

            pd.DataFrame(t_list_comp).to_csv(os.path.join(rp.results_save_string, 'parameter_hypothesis_testing.csv'))
            pd.DataFrame(t_list_prop).to_csv(os.path.join(rp.results_save_string, 'rationality_proportion_hypothesis_testing.csv'))
        except FileNotFoundError:
            try:
                os.makedirs(os.path.join(rp.results_save_string))
                with open(os.path.join(rp.results_save_string, 'choice_analysis.csv'), 'w') as f:
                    df_choices.to_csv(f)
            except FileExistsError as e:
                print(e)

    return df_choices, pd.DataFrame(t_list_comp), pd.DataFrame(t_list_prop)


_ = create_output(rp=ResultsParams(results_param_dict))


# def analyze():
#     # ============== TEST OF DIFFERENCES ==============
#     from assumption_calc_functions import two_sample_test_ind_means
#     from assumption_calc_functions import two_sample_test_prop
#
#     # specify players and cases
#     players_for_testing = ['pluribus',  'eddie_mrorange_joe_mrblonde_gogo_bill_mrpink_oren_mrblue_budd_mrbrown_mrwhite_hattori']  #, 'eddie', 'bill', 'eddie_mrorange_joe_mrblonde_gogo_bill_mrpink_oren_mrblue_budd_mrbrown_mrwhite_hattori'] # 'mrpink', 'mrorange', 'bill',
#     cases_for_testing = ['post_neutral_or_blind_only', 'post_win_excl_blind_only'] # must only be a list of two currently,  'post_loss_excl_blind_only',
#     params_for_testing = ['omega', 'lambda']
#     prob_case_for_testing = 'dnn_prob'
#
#     # load relevant data
#     results_dict = {p: {c: {} for c in cases_for_testing} for p in players_for_testing}
#     for p in players_for_testing:
#         for c in cases_for_testing:
#             with open(os.path.join('output', 'iter_multistart_saves', p, prob_case_for_testing, 'est_params', p + '_' + c + '.json'), 'r') as f:
#                 results_dict[p][c] = json.load(f)
#         del c
#     del p
#
#     # compare parameter estimates by case
#     print('\n\n===== change in RISK =====')
#     print('For two sample test of independent means (expected value of estimated parameters)')
#     print('case 1: %s, \t case2: %s' % (cases_for_testing[0], cases_for_testing[1]))
#     for p in players_for_testing:
#         print('\n\n\tFor player %s:' % p)
#         for param in params_for_testing:
#             print('\t\tParameter: %s' % param)
#             print('\t\t\tCase 1: %s, mean = %3.2f, stdev = %3.2f, n = %d' % (cases_for_testing[0], results_dict[p][cases_for_testing[0]][param]['mean'], results_dict[p][cases_for_testing[0]][param]['stdev'], results_dict[p][cases_for_testing[0]][param]['nobs']))
#             print('\t\t\tCase 2: %s, mean = %3.2f, stdev = %3.2f, n = %d' % (cases_for_testing[1], results_dict[p][cases_for_testing[1]][param]['mean'], results_dict[p][cases_for_testing[1]][param]['stdev'], results_dict[p][cases_for_testing[1]][param]['nobs']))
#             print('\t\tt-stat: %3.2f, p-value: %3.1f' % (two_sample_test_ind_means(results_dict[p][cases_for_testing[0]][param]['mean'], results_dict[p][cases_for_testing[1]][param]['mean'],
#                                                                                 results_dict[p][cases_for_testing[0]][param]['stdev'], results_dict[p][cases_for_testing[1]][param]['stdev'],
#                                                                                      results_dict[p][cases_for_testing[0]][param]['nobs'], results_dict[p][cases_for_testing[1]][param]['nobs'], n_sides=2, print_f=False)))
#         del param
#     del p
#
#     # compare proportion of rational decisions by case
#     print('\n\n===== change in RATIONALITY ======')
#     print('For two sample test of proportions (change in proportion of rational actions)')
#     print('Player 1: %s, \t Player 2: %s' % (players_for_testing[0], players_for_testing[1]))
#     for case in cases_for_testing:
#         print('\tFor case %s:' % case)
#         n1 = sum([v for k, v in results_dict[players_for_testing[0]][case].items() if k in ['rational_play', 'rational_fold', 'irrational_play', 'irrational_fold']])
#         p1 = (results_dict[players_for_testing[0]][case]['rational_play'] + results_dict[players_for_testing[0]][case]['rational_fold'])/n1
#         n2 = sum([v for k, v in results_dict[players_for_testing[1]][case].items() if k in ['rational_play', 'rational_fold', 'irrational_play', 'irrational_fold']])
#         p2 = (results_dict[players_for_testing[1]][case]['rational_play'] + results_dict[players_for_testing[1]][case]['rational_fold'])/n2
#
#         print('\t\tproportion = %3.2f, n = %d, player 1: %s' % (p1, n1, players_for_testing[0]))
#         print('\t\tproportion = %3.2f, n = %d, player 2: %s' % (p2, n2, players_for_testing[1]))
#         print('\tt-stat: %3.2f, p-value: %3.3f' % (two_sample_test_prop(p1, p2, n1, n2,
#                                                      n_sides_f=2)))
#     del case
#
#     # compare proportion of rational actions across players
#     print('\n\n===== change in RATIONALITY ======')
#     print('For two sample test of proportions (change in proportion of rational actions)')
#     print('case 1: %s, \t case2: %s' % (cases_for_testing[0], cases_for_testing[1]))
#     for p in players_for_testing:
#         print('\tFor player %s:' % p)
#         n1 = sum([v for k, v in results_dict[p][cases_for_testing[0]].items() if
#                   k in ['rational_play', 'rational_fold', 'irrational_play', 'irrational_fold']])
#         p1 = (results_dict[p][cases_for_testing[0]]['rational_play'] + results_dict[p][cases_for_testing[0]][
#             'rational_fold']) / n1
#         n2 = sum([v for k, v in results_dict[p][cases_for_testing[1]].items() if
#                   k in ['rational_play', 'rational_fold', 'irrational_play', 'irrational_fold']])
#         p2 = (results_dict[p][cases_for_testing[1]]['rational_play'] + results_dict[p][cases_for_testing[1]][
#             'rational_fold']) / n2
#
#         print('\t\tCase 1: %s, proportion = %3.2f, n = %d' % (cases_for_testing[0], p1, n1))
#         print('\t\tCase 2: %s, proportion = %3.2f, n = %d' % (cases_for_testing[1], p2, n2))
#         print('\tt-stat: %3.2f, p-value: %3.3f' % (two_sample_test_prop(p1, p2, n1, n2,
#                                                                         n_sides_f=2)))


# analyze()


# -------- UPDATING CHOICE SITUATIONS TO ADD ID, CAN DELETE -----------------
# cp = ControlParams(control_param_dict)
#
# for player in ['Bill', 'Eddie', 'Joe', 'MrOrange', 'MrPink', 'Pluribus']:
# player = ['Eddie', 'MrOrange', 'Joe', 'MrBlonde', 'Gogo', 'Bill', 'MrPink', 'Oren', 'MrBlue', 'Budd', 'MrBrown', 'MrWhite', 'Hattori']    # 'eddie_mrorange_']
# cp.select_player_list = player
# cp.create_dict_params_to_set()
# print(cp.select_player_list_save_string)
#
# with open(os.path.join(cp.choice_situations_dir_save_string, cp.select_player_list_save_string), 'rb') as f:
#     choice_situations = pickle.load(f)
#
# # new_cs = list()
# for cs in choice_situations:
#     cs.tags.update({'id': binascii.b2a_hex(os.urandom(4))})
#     # t_cs = ChoiceSituation(sit_options=cs.options)
#     # for k, v in cs.__dict__.items():
#     #     print(k, v)
#     #     t_cs.__setattr__(k, v)
#
# for cs in choice_situations:
#     print('%s %s' % (cs.tags['id'], cs.omega_indifference))
#
# with open(os.path.join(cp.choice_situations_dir_save_string, cp.select_player_list_save_string), 'wb') as f:
#     pickle.dump(choice_situations, f)