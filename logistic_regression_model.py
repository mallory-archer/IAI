import math
import pickle
import pandas as pd
from statsmodels.api import Logit
import copy
from assumption_calc_functions import two_sample_test_prop

pd.options.display.max_columns = 25


class LogisticRegression:
    def __init__(self, endog_name_f=None, exog_name_f=None, data_f=None, add_constant_f=True, scale_vars_list_f=list(),
                 interaction_name_f=list(), convert_bool_dict_f=dict(), convert_ord_list_f=list(), cat_col_omit_dict_f=dict(), **kwds):
        self.endog_name = endog_name_f
        self.exog_name = exog_name_f
        self.data = data_f.reindex()
        self.add_constant = add_constant_f
        self.interaction_name = interaction_name_f
        self.convert_bool_dict = convert_bool_dict_f   # convert_bool_dict_f
        self.convert_ord_list = convert_ord_list_f    # convert_ord_list_f
        self.cat_col_names = list()
        self.cat_col_omit_dict = cat_col_omit_dict_f
        self.cat_col_drop_names = list()
        self.dummy_col_omit_list = list()
        self.scale_vars_list = scale_vars_list_f
        self.exog_name_model = None
        self.model_data = None
        self.model = None
        self.model_result = None

        self.refresh_model_data()

    def check_for_exog_conflict(self):
        t_bool_ord = set(self.convert_bool_dict.keys()).intersection(set(self.convert_ord_list))
        t_cat_bool = set(self.cat_col_omit_dict.keys()).intersection(set(self.convert_bool_dict.keys()))
        t_cat_ord = set(self.cat_col_omit_dict.keys()).intersection(set(self.convert_ord_list))

        if len(t_bool_ord) > 0:
            print('WARNING appearing in both boolean and ordinal variable lists: %s' % ', '.join(t_bool_ord))
        if len(t_cat_ord) > 0:
            print('WARNING appearing in both categorical and ordinal variable lists: %s, ignoring categorical' % ', '.join(t_cat_ord))
        if len(t_cat_bool) > 0:
            print('WARNING appearing in both categorical and boolean variable lists: %s, ignoring categorical' % ', '.join(t_cat_bool))

    def convert_cat_to_dummies(self):
        # get list of exogenous variables that are categorical and need to be converted
        self.cat_col_names = [x for x in self.exog_name if
                              ((x not in list(self.convert_bool_dict.keys())) and
                               (x not in self.convert_ord_list) and
                               (self.data[x].dtype == 'O'))]
        prefix_sep = '_'
        [self.cat_col_omit_dict.update({x: self.data[x].mode(dropna=True).values[0]}) for x in self.cat_col_names if x not in list(self.cat_col_omit_dict.keys())]
        self.cat_col_drop_names = [k + prefix_sep + v for k, v in self.cat_col_omit_dict.items()]

        if len(self.cat_col_names) > 0:
            return pd.get_dummies(self.data[self.cat_col_names], prefix_sep=prefix_sep, columns=self.cat_col_names,
                                  dtype=bool)
        else:
            return None

    def convert_to_bool(self):
        t_df = pd.DataFrame()
        t_col_names = list()
        for k, v in self.convert_bool_dict.items():
            t_col_names.append(k + '_' + v + '_TF')
            t_df = pd.concat([t_df, self.data[k] == v], axis=1)
        t_df.columns = t_col_names
        return t_df

    def convert_to_ordinal(self):
        t_df = pd.DataFrame()
        t_col_names = list()
        for c in self.convert_ord_list:
            t_col_names.append(c + '_ORD')
            t_df = pd.concat([t_df, self.data[c].astype(int)], axis=1)
        t_df.columns = t_col_names
        return t_df

    def create_interactions(self):
        def create_dummy_df(data_f, v1, v2, drop_list_f):
            prefix_sep = '_'

            if (data_f[v1].dtype == bool) and (data_f[v2].dtype == bool):
                # both bool - create interaction effect directly
                t_df = pd.DataFrame(data_f[v1] & data_f[v2], columns=[v1 + ' * ' + v2 + '_INT'])
                return t_df, ({v1: None}, {v2: None})
            elif (data_f[v1].dtype != bool) and (data_f[v2].dtype != bool):
                # both cat
                v1_dummies = pd.get_dummies(data_f[v1], prefix_sep=prefix_sep, dtype=bool)
                v1_omit = data_f[v1].mode(dropna=True).values[0] if v1 not in list(
                    drop_list_f.keys()) else drop_list_f[v1]

                v2_dummies = pd.get_dummies(data_f[v2], prefix_sep=prefix_sep, dtype=bool)
                v2_omit = data_f[v2].mode(dropna=True).values[0] if v2 not in list(
                    drop_list_f.keys()) else drop_list_f[v2]
                t_df = pd.DataFrame(index=data_f.index)
                for c1 in [x for x in v1_dummies.columns if x != v1_omit]:
                    for c2 in [x for x in v2_dummies.columns if x != v2_omit]:
                        t_df = pd.concat(
                            [t_df, pd.DataFrame(v1_dummies[c1] & v2_dummies[c2], columns=[c1 + ' * ' + c2 + '_INT'])],
                            axis=1)
                return t_df, ({v1: v1_omit}, {v2: v2_omit})
            else:
                # one bool
                if data_f[v1].dtype == bool:
                    vb = v1
                    vd = v2
                else:
                    vb = v2
                    vd = v1
                vd_dummies = pd.get_dummies(data_f[vd], prefix_sep=prefix_sep, dtype=bool)
                vd_omit = data_f[vd].mode(dropna=True).values[0] if vd not in list(
                    drop_list_f.keys()) else drop_list_f[vd]
                t_df = pd.DataFrame(index=data_f.index)
                for c in [x for x in vd_dummies.columns if x != vd_omit]:
                    t_df = pd.concat(
                        [t_df, pd.DataFrame(data_f[vb] & vd_dummies[c], columns=[vb + ' * ' + c + '_INT'])], axis=1)
                return t_df, ({vb: None}, {vd: None})

        t_df = pd.DataFrame(index=self.data.index)
        t_dummy_col_omit_list = list()
        for int_act_col1, int_act_col2 in self.interaction_name:
            t_dummy, t_dummy_omit = create_dummy_df(self.data, int_act_col1, int_act_col2, self.cat_col_omit_dict)
            t_df = pd.concat([t_df, t_dummy], axis=1)
            t_dummy_col_omit_list.append(t_dummy_omit)
            del t_dummy, t_dummy_omit
        del int_act_col1, int_act_col2
        self.dummy_col_omit_list = t_dummy_col_omit_list
        
        return t_df
        
    def code_variables(self):
        # get new variable matrices
        if len(self.convert_bool_dict) > 0:
            df_bool_f = self.convert_to_bool()
        else:
            df_bool_f = None

        if len(self.convert_ord_list) > 0:
            df_ord_f = self.convert_to_ordinal()
        else:
            df_ord_f = None

        df_cat_f = self.convert_cat_to_dummies()

        if len(self.interaction_name) > 0:
            df_interaction_f = self.create_interactions()
        else:
            df_interaction_f = None

        return df_bool_f, df_ord_f, df_cat_f, df_interaction_f

    def refresh_model_data(self):
        df_bool_f, df_ord_f, df_cat_f, df_interaction_f = self.code_variables()

        self.check_for_exog_conflict()

        t_remain_exog = [x for x in self.exog_name if ((x not in list(self.convert_bool_dict.keys())) and
                                                       (x not in list(self.convert_ord_list)) and
                                                       (x not in self.cat_col_names))]

        if df_cat_f is not None:
            df_cat_f_dropped_omit = df_cat_f[[c for c in df_cat_f.columns if c not in self.cat_col_drop_names]]
        else:
            df_cat_f_dropped_omit = None

        self.model_data = pd.concat([self.data[self.endog_name], self.data[t_remain_exog], df_bool_f, df_ord_f, df_cat_f_dropped_omit, df_interaction_f], axis=1)

        self.exog_name_model = [x for x in self.model_data if x != self.endog_name]

    def create_model_object(self):
        model_mat = copy.deepcopy(self.model_data)

        # convert booleans to floats explicitly
        for c in model_mat.columns:
            if model_mat[c].dtype == bool:
                model_mat[c] = model_mat[c].astype(float)

        # scale specified vars to N(0,1)
        for c in self.scale_vars_list:
            try:
                xbar = model_mat[c].mean()
                s = model_mat[c].std()
                model_mat[c] = model_mat[c].apply(lambda x: (x - xbar) / s)
                del xbar, s
            except KeyError:
                print('Warning: specified variable to scale, %s, is not included in model covariates' % c)

        # drop rows with na
        model_mat.dropna(inplace=True)

        # add constant if needed
        if self.add_constant:
            model_mat = pd.concat([pd.DataFrame(data=[1]*model_mat.shape[0], index=model_mat.index, columns=['const']), model_mat], axis=1)

        self.model = Logit(endog=model_mat[self.endog_name], exog=model_mat[[c for c in model_mat.columns if c != self.endog_name]])

    def estimate_model(self):
        self.refresh_model_data()
        self.create_model_object()
        self.model_result = self.model.fit()
        print(self.model_result.summary())


def create_master_data_frame(games_f):
    # configure data frame
    obs_list_f = list()
    for g_num, g in games_f.items():
        for h_num, h in g.hands.items():
            for p_name in h.players:
                t_dict = {'game': g_num, 'hand': h_num, 'player': p_name,
                          'slansky': h.odds[p_name]['slansky'], 'seat': int(h.players.index(p_name)) + 1,
                          'stack_rank': h.start_stack_rank[p_name], 'start_stack': h.start_stack[p_name],
                          'preflop_action': h.actions['preflop'][p_name], 'outcome': h.outcomes[p_name]}
                try:
                    t_prev_outcome = g.hands[str(int(h_num) - 1)].outcomes[p_name]
                    t_relative_start_stack = h.relative_start_stack[p_name]
                except KeyError:
                    t_prev_outcome = None
                    t_relative_start_stack = None
                t_dict.update({'outcome_previous': t_prev_outcome, 'relative_start_stack': t_relative_start_stack})
                obs_list_f.append(t_dict)
                del t_dict
    df_f = pd.DataFrame(obs_list_f)

    # specify type
    df_f = df_f.astype({'game': str, 'hand': str, 'player': str,
                        'slansky': str, 'seat': str, 'stack_rank': str, 'start_stack': float,
                        'relative_start_stack': float,
                        'preflop_action': str, 'outcome': float, 'outcome_previous': float,
                        })
    return df_f


def engineer_features(df_f):
    # add additional features
    df_f['preflop_fold_TF'] = (df_f['preflop_action'] == 'f')
    df_f['human_player_TF'] = (df_f['player'] != 'Pluribus')
    
    # categorize loss, win, neutral
    df_f['outcome_previous_cat'] = 'neutral'
    df_f.loc[df_f['outcome_previous'] < 0, 'outcome_previous_cat'] = 'loss'
    df_f.loc[df_f['outcome_previous'] > 0, 'outcome_previous_cat'] = 'win'
    df_f['loss_outcome_previous_TF'] = (df_f['outcome_previous_cat'] == 'loss')
    df_f['win_outcome_previous_TF'] = (df_f['outcome_previous_cat'] == 'win')

    df_f['zero_outcome_previous_TF'] = (df_f.outcome_previous == 0)  # fold, no blind ("zero" outcome_previous)
    df_f['blind_only_outcome_previous_TF'] = (abs(df_f.outcome_previous) == 50) | (abs(df_f.outcome_previous) == 100) | (abs(df_f.outcome_previous) == 150)
    df_f['zero_or_blind_only_outcome_previous_TF'] = df_f['zero_outcome_previous_TF'] | df_f['blind_only_outcome_previous_TF']
    df_f['loss_outcome_xonlyblind_previous_TF'] = df_f['loss_outcome_previous_TF'] & (df_f.outcome_previous != -50) & (df_f.outcome_previous != -100)
    df_f['win_outcome_xonlyblind_previous_TF'] = df_f['win_outcome_previous_TF'] & (df_f.outcome_previous != 150)

    # grouping players (somewhat arbitrary...)
    df_f['sneaky_robot_player_TF'] = df_f['player'].apply(lambda x: x in ['Bill', 'MrBrown', 'MrPink', 'MrWhite'])
    df_f = df_f.astype({'preflop_fold_TF': bool, 'human_player_TF': bool,
                        'outcome_previous_cat': str, 'loss_outcome_previous_TF': bool, 'win_outcome_previous_TF': bool,
                        'zero_outcome_previous_TF': bool, 'blind_only_outcome_previous_TF': bool,
                        'zero_or_blind_only_outcome_previous_TF': bool,
                        'loss_outcome_xonlyblind_previous_TF': bool, 'win_outcome_xonlyblind_previous_TF': bool
                        })
    return df_f


def print_df_summary(df_f, return_player_summary_f=True):
    df_player_summary_f = pd.concat(
        [df_f['player'].value_counts(), df_f.loc[df_f['outcome_previous'].isnull(), 'player'].value_counts(),
         df_f.groupby('player')['preflop_fold_TF'].sum().astype(int) / df_f['player'].value_counts()], axis=1)
    df_player_summary_f.columns = ['num. hands', 'num hands missing previous action', '% hands folded preflop']

    t_df_wins = df_f.loc[df_f['outcome'] > 0]
    print('\nDATAFRAME SUMMARY')
    print('%d total obs' % df_f.shape[0])
    print('%d games, %d hands, %3.0f avg. hands / game'  % (df_f.game.nunique(), df_f[['game', 'hand']].groupby('game').nunique().sum()['hand'], df_f[['game', 'hand']].groupby('game').nunique()['hand'].mean()))
    print('%d min. players per game, max. %d players per game' % (df_f.groupby('game')['player'].nunique().min(), df_f.groupby('game')['player'].nunique().max()))
    print('winnings per hand: $%3.0f min., $%3.0f max., $%3.0f avg., $%3.0f med.' % (t_df_wins.groupby(['game', 'hand'])['outcome'].sum().min(),
                                                                                     t_df_wins.groupby(['game', 'hand'])['outcome'].sum().max(),
                                                                                     t_df_wins.groupby(['game', 'hand'])['outcome'].sum().mean(),
                                                                                     t_df_wins.groupby(['game', 'hand'])['outcome'].sum().median()))
    print('all players: %s' % ', '.join(list(df_f['player'].unique())))
    print('\nOBS (HANDS) PLAYER SUMMARY:')
    print(df_player_summary_f)
    if return_player_summary_f:
        return df_player_summary_f


def create_formatted_output(df_f, cases_f):
    # summary statistics fold/play
    def create_df_row(df_ff):
        t_df_ff = df_ff.groupby('human_player_TF').preflop_fold_TF.agg(['count', 'sum'])
        t_df_ff['perc'] = t_df_ff['sum']/t_df_ff['count']
        return {'human_perc_preflop_fold': t_df_ff.loc[True, 'perc'], 'human_nobs': t_df_ff.loc[True, 'count'], 'ADM_perc_preflop_fold': t_df_ff.loc[False, 'perc'], 'ADM_nobs': t_df_ff.loc[False, 'count']}

    df_output_f = pd.DataFrame.from_dict({'all': create_df_row(df_f)}, orient='index')
    for filter in cases_f:
        df_output_f = pd.concat([df_output_f, pd.DataFrame.from_dict({filter: create_df_row(df_f.loc[df_f[filter]])}, orient='index')], axis=0)

    return df_output_f


def run_within_hypothesis_tests(df_f, n_sides_f, baseline_name_f, test_case_names_f, player_names_f):
    # within-player hypothesis tests for diff in probability of folding preflop (do players react to losing/winning?)
    col_name1_f = '_pval_2sample_vs_'
    col_name2_f = '_diff_'
    for p in player_names_f:
        df_f[p + col_name2_f + baseline_name_f] = None
        df_f[p + col_name1_f + baseline_name_f] = None
        for test_case in test_case_names_f:
            if test_case != baseline_name_f:
                df_f.loc[test_case, p + col_name2_f + baseline_name_f] = df_f.loc[test_case, p + '_perc_preflop_fold'] - df_f.loc[baseline_name_f, p + '_perc_preflop_fold']
                _, df_f.loc[test_case, p + col_name1_f + baseline_name_f] = two_sample_test_prop(
                    df_f.loc[baseline_name_f, p + '_perc_preflop_fold'],
                    df_f.loc[test_case, p + '_perc_preflop_fold'],
                    df_f.loc[baseline_name_f, p + '_nobs'],
                    df_f.loc[baseline_name_f, p + '_nobs'], n_sides_f)
    return df_f


def inverse_logit(x):
    return math.exp(x) / (1 + math.exp(x))


def format_logistic_regression_output(logistic_model_f):
    print('WARNING: calculation of mean effects assumes CONSTANT was present in model.')
    # get mean of coefficient values to calculate average effects

    coef_const_vals_f = dict(logistic_model_f.model_data[[x for x in logistic_model_f.model_result.params.index
                                                          if ((x != 'const') and
                                                          (logistic_model_f.model_data[x].dtype != bool))]].mean())
    coef_const_vals_f.update({'const': 1})
    coef_const_vals_f.update(dict([(x, 0) for x in logistic_model_f.model_result.params.index if ((x != 'const') and (logistic_model_f.model_data[x].dtype == bool))]))

    df_f = pd.DataFrame(data={'params': logistic_model_f.model_result.params, 'std_err': logistic_model_f.model_result.bse, 't_stat': logistic_model_f.model_result.tvalues, 'pvalue': logistic_model_f.model_result.pvalues, 'base_value': coef_const_vals_f})
    reg_col_names_f = list(df_f.columns)
    df_f.loc['const', 'inverse_logit_incl_base'] = inverse_logit(sum([df_f.loc[k, 'params'] * v for k, v in coef_const_vals_f.items()]))
    df_f.loc['const', 'increase_prob_from_base'] = df_f.loc['const', 'inverse_logit_incl_base']
    for label in df_f.index:
        if label != 'const':
            df_f.loc[label, 'inverse_logit_incl_base'] = inverse_logit(sum([df_f.loc[k, 'params'] * v for k, v in coef_const_vals_f.items()]) + df_f.loc[label, 'params'])
            df_f.loc[label, 'increase_prob_from_base'] = df_f.loc['const', 'inverse_logit_incl_base'] - df_f.loc[label, 'inverse_logit_incl_base']

    label_order_f = ['const'] + [x for x in df_f.index if (x.find(' * ') == -1) & (x != 'const')] + [x for x in df_f.index if x.find(' * ') > -1]
    return df_f.loc[label_order_f, reg_col_names_f + ['increase_prob_from_base', 'inverse_logit_incl_base']]


# ----- LOAD DATA -----
with open("python_hand_data.pickle", 'rb') as f:
    data = pickle.load(f)
players = data['players']
games = data['games']

# ----- CREATE DATAFRAME -----
df_master = create_master_data_frame(games)
df_master = engineer_features(df_master)
print_df_summary(df_master, return_player_summary_f=False)

# ----- HYPOTHESIS TESTING -----
df_data_summary = create_formatted_output(df_master, cases_f=['zero_outcome_previous_TF',
                                                              'loss_outcome_previous_TF', 'win_outcome_previous_TF',
                                                              'zero_or_blind_only_outcome_previous_TF',
                                                              'loss_outcome_xonlyblind_previous_TF',
                                                              'win_outcome_xonlyblind_previous_TF',
                                                              'blind_only_outcome_previous_TF'
                                                              ])
hyp_test_specs = {'test_1': {'baseline': 'zero_outcome_previous_TF',
                             'test_cases': ['loss_outcome_previous_TF', 'win_outcome_previous_TF']},
                  'test_2': {'baseline': 'zero_or_blind_only_outcome_previous_TF',
                             'test_cases': ['loss_outcome_xonlyblind_previous_TF', 'win_outcome_xonlyblind_previous_TF']}}
for specs in hyp_test_specs.values():
    df_data_summary = run_within_hypothesis_tests(df_data_summary, n_sides_f=2,
                                              baseline_name_f=specs['baseline'],
                                              test_case_names_f=specs['test_cases'],
                                              player_names_f=['human', 'ADM'])

# df_data_summary.to_csv('data_summary.csv')

# ----- SPECIFY MODEL PARAMS ----
endog_var_name = 'preflop_fold_TF'
add_constant = True
exog_var_name = []    # 'player', 'loss_outcome_xonlyblind_previous_TF', 'win_outcome_xonlyblind_previous_TF',
ordinal_vars = ['slansky', 'seat_map']    # 'seat_map': seat_map reassigns the first seat to the player sitting in seat 3; this is to allow for ordinal varaibles to account for nonlinearity in seats 1 and 2 as a result of blinds
bool_vars = {}
categorical_drop_vals = {'seat': '3', 'stack_rank': '1'}    # 'player': 'Pluribus',
interaction_vars = [('loss_outcome_xonlyblind_previous_TF', 'human_player_TF'), ('win_outcome_xonlyblind_previous_TF', 'human_player_TF')]   # ('loss_outcome_xonlyblind_previous_TF', 'player'), ('win_outcome_xonlyblind_previous_TF', 'player')
scale_vars_list = ['start_stack']

# filter and map
df_data = df_master.reindex()
df_data['seat_map'] = df_data['seat'].map({'1': '5', '2': '6', '3': '1', '4': '2', '5': '3', '6': '4'})

logistic_model = LogisticRegression(endog_name_f=endog_var_name, exog_name_f=exog_var_name, data_f=df_data,
                                    add_constant_f=add_constant, scale_vars_list_f=scale_vars_list,
                                    convert_ord_list_f=ordinal_vars, convert_bool_dict_f=bool_vars,
                                    cat_col_omit_dict_f=categorical_drop_vals, interaction_name_f=interaction_vars)
logistic_model.estimate_model()

df_coef_output = format_logistic_regression_output(logistic_model)

# df_coef_output.to_csv('logistic_regression_output.csv')
