import json
import copy
import numpy as np
import pandas as pd
import random
import os
import pickle

import matplotlib.pyplot as plt

from RUM_functions_classes import RandomUtilityModel, ChoiceSituation, Option
from RUM_functions_classes import generate_choice_situations, generate_synthetic_data, reformat_choice_situations_for_model, calc_CRRA_utility, run_multistart, parse_multistart, calc_robust_varcov, calc_mle_tstat, check_likelihood
from assumption_calc_functions import two_sample_test_prop
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
                    'play': t_play,
                    'fold': t_fold,
                    'rational_TF': t_rational_TF,
                    'rational_choice_type': t_type}})
    return choice_param_dictionary, 'exp_util_omega_' + str(int(round(exp_omega, 4) * 10000)) + 'e4'


def count_rational_action_types(choice_param_dictionary, exp_omega_key):
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
            except FileExistsError as e:
                print(e)


def main(cp):
    # (REQ) ===== SET CONTROL PARAMS =====
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
    #                         select_plot_cases=['post_neutral_or_blind_only', 'post_loss_excl_blind_only'], plot_TF=True)

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
    select_results = run_multistart(nstart_points=cp.num_multistarts, est_param_dictionary={'options': {'ftol': cp.ftol, 'gtol': cp.gtol, 'maxiter': cp.maxiter, 'disp': None}, 'lb': cp.lb_model, 'ub': cp.ub_model}, model_object=model, save_iter=cp.save_iter, save_path=cp.multi_start_dir_save_string, save_index_start=cp.save_index_start)

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
    for player in rp.select_player_list:
        if not isinstance(player, list):
            rp.set_player_list_save_string([player])
        else:
            rp.set_player_list_save_string(player)
        print(rp.select_player_list_save_string)
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
                with open(os.path.join(rp.param_estimates_dir_save_string,
                                       rp.select_player_list_save_string + '_' + state + '.json'), 'r') as f:
                    estimates_dict = json.load(f)

                    choice_param_dictionary, exp_omega_key = calc_util_rational(estimates_dict['omega']['mean'],
                                                                                choice_param_dictionary)

                for rank in choice_param_dictionary.keys():
                    for seat in choice_param_dictionary[rank].keys():
                        for item in choice_param_dictionary[rank][seat]:
                            # item = choice_param_dictionary[1][1][0] #######
                            t_dict = {'CRRA_gamble_type': item['CRRA_gamble_type'], 'player': player, 'seat': seat,
                                      'rank': rank, 'cs_id': item['id']}
                            for k, outcomes in item['params'].items():
                                for i in range(len(outcomes)):
                                    t_dict.update({k + '_' + j + str(i): v for j, v in outcomes[i].items()})
                            t_dict.update(item['n_chosen'])
                            for k in ['rational_TF', 'rational_choice_type']:
                                t_dict.update({k: item[exp_omega_key][k]})
                            t_dict.update({'state': [x for x in rp.select_states if [cs for cs in choice_situations if
                                                                                     (cs.slansky_strength == rank) and (
                                                                                             cs.seat == seat) and (
                                                                                             cs.tags['id'] == item[
                                                                                         'id'])][
                                0].__getattribute__(x)][0]})
                            t_obs_list.append(t_dict)
                            del t_dict
                        del item
                    del seat
                del rank

            except FileNotFoundError as e:
                print('WARNING: %s' % e)

            del choice_param_dictionary
        del state
    del player

    # create data frame for export
    df_choices = pd.DataFrame(t_obs_list)
    df_choices['choice'] = 'fold'
    df_choices.loc[df_choices['play'] == 1, 'choice'] = 'play'
    df_choices.loc[df_choices['fold'] == 1, 'choice'] = 'fold'
    if any(df_choices.loc[df_choices['play'] == 1, 'choice'] != 'play') or any(
            df_choices.loc[df_choices['fold'] == 1, 'choice'] != 'fold'):
        print('WARNING: mismapping of play/fold binary columns')

    if rp.save_TF:
        try:
            with open(os.path.join(rp.results_save_string, 'choice_analysis.csv'), 'w') as f:
                df_choices.to_csv(f)
        except FileNotFoundError:
            try:
                os.makedirs(os.path.join(rp.results_save_string))
                with open(os.path.join(rp.results_save_string, 'choice_analysis.csv'), 'w') as f:
                    df_choices.to_csv(f)
            except FileExistsError as e:
                print(e)

    return df_choices


_ = create_output(rp=ResultsParams(results_param_dict))


def analyze():
    # ============== TEST OF DIFFERENCES ==============
    from assumption_calc_functions import two_sample_test_ind_means
    from assumption_calc_functions import two_sample_test_prop

    # specify players and cases
    players_for_testing = ['pluribus',  'eddie_mrorange_joe_mrblonde_gogo_bill_mrpink_oren_mrblue_budd_mrbrown_mrwhite_hattori']  #, 'eddie', 'bill', 'eddie_mrorange_joe_mrblonde_gogo_bill_mrpink_oren_mrblue_budd_mrbrown_mrwhite_hattori'] # 'mrpink', 'mrorange', 'bill',
    cases_for_testing = ['post_neutral_or_blind_only', 'post_win_excl_blind_only'] # must only be a list of two currently,  'post_loss_excl_blind_only',
    params_for_testing = ['omega', 'lambda']
    prob_case_for_testing = 'dnn_prob'

    # load relevant data
    results_dict = {p: {c: {} for c in cases_for_testing} for p in players_for_testing}
    for p in players_for_testing:
        for c in cases_for_testing:
            with open(os.path.join('output', 'iter_multistart_saves', p, prob_case_for_testing, 'est_params', p + '_' + c + '.json'), 'r') as f:
                results_dict[p][c] = json.load(f)
        del c
    del p

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
        del param
    del p

    # compare proportion of rational decisions by case
    print('\n\n===== change in RATIONALITY ======')
    print('For two sample test of proportions (change in proportion of rational actions)')
    print('Player 1: %s, \t Player 2: %s' % (players_for_testing[0], players_for_testing[1]))
    for case in cases_for_testing:
        print('\tFor case %s:' % case)
        n1 = sum([v for k, v in results_dict[players_for_testing[0]][case].items() if k in ['rational_play', 'rational_fold', 'irrational_play', 'irrational_fold']])
        p1 = (results_dict[players_for_testing[0]][case]['rational_play'] + results_dict[players_for_testing[0]][case]['rational_fold'])/n1
        n2 = sum([v for k, v in results_dict[players_for_testing[1]][case].items() if k in ['rational_play', 'rational_fold', 'irrational_play', 'irrational_fold']])
        p2 = (results_dict[players_for_testing[1]][case]['rational_play'] + results_dict[players_for_testing[1]][case]['rational_fold'])/n2

        print('\t\tproportion = %3.2f, n = %d, player 1: %s' % (p1, n1, players_for_testing[0]))
        print('\t\tproportion = %3.2f, n = %d, player 2: %s' % (p2, n2, players_for_testing[1]))
        print('\tt-stat: %3.2f, p-value: %3.3f' % (two_sample_test_prop(p1, p2, n1, n2,
                                                     n_sides_f=2)))
    del case

    # compare proportion of rational actions across players
    print('\n\n===== change in RATIONALITY ======')
    print('For two sample test of proportions (change in proportion of rational actions)')
    print('case 1: %s, \t case2: %s' % (cases_for_testing[0], cases_for_testing[1]))
    for p in players_for_testing:
        print('\tFor player %s:' % p)
        n1 = sum([v for k, v in results_dict[p][cases_for_testing[0]].items() if
                  k in ['rational_play', 'rational_fold', 'irrational_play', 'irrational_fold']])
        p1 = (results_dict[p][cases_for_testing[0]]['rational_play'] + results_dict[p][cases_for_testing[0]][
            'rational_fold']) / n1
        n2 = sum([v for k, v in results_dict[p][cases_for_testing[1]].items() if
                  k in ['rational_play', 'rational_fold', 'irrational_play', 'irrational_fold']])
        p2 = (results_dict[p][cases_for_testing[1]]['rational_play'] + results_dict[p][cases_for_testing[1]][
            'rational_fold']) / n2

        print('\t\tCase 1: %s, proportion = %3.2f, n = %d' % (cases_for_testing[0], p1, n1))
        print('\t\tCase 2: %s, proportion = %3.2f, n = %d' % (cases_for_testing[1], p2, n2))
        print('\tt-stat: %3.2f, p-value: %3.3f' % (two_sample_test_prop(p1, p2, n1, n2,
                                                                        n_sides_f=2)))


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