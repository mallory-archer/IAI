import pickle
import pandas as pd
from statsmodels.api import Logit
import copy
from assumption_calc_functions import two_sample_test_prop

pd.options.display.max_columns = 25


class LogisticRegression:
    def __init__(self, endog_name_f=None, exog_name_f=None, data_f=None, add_constant_f=True,
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
                except KeyError:
                    t_prev_outcome = None
                t_dict.update({'outcome_previous': t_prev_outcome})
                obs_list_f.append(t_dict)
                del t_dict
    df_f = pd.DataFrame(obs_list_f)

    # specify type
    df_f = df_f.astype({'game': str, 'hand': str, 'player': str,
                        'slansky': str, 'seat': str, 'stack_rank': str, 'start_stack': float,
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


# print(create_formatted_output(df_master))

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
                                                              'loss_outcome_xonlyblind_previous_TF',
                                                              'win_outcome_xonlyblind_previous_TF',
                                                              'blind_only_outcome_previous_TF',
                                                              'zero_or_blind_only_outcome_previous_TF'
                                                              ])
df_data_summary = run_within_hypothesis_tests(df_data_summary, n_sides_f=2,
                                              baseline_name_f='zero_outcome_previous_TF',
                                              test_case_names_f=['loss_outcome_previous_TF', 'win_outcome_previous_TF',
                                                                 'zero_outcome_previous_TF', 'blind_only_outcome_previous_TF',
                                                                 'zero_or_blind_only_outcome_previous_TF',
                                                                 'loss_outcome_xonlyblind_previous_TF',
                                                                 'win_outcome_xonlyblind_previous_TF'],
                                              player_names_f=['human', 'ADM'])

df_data_summary.to_csv('data_summary.csv')

# ----- SPECIFY MODEL PARAMS ----
endog_var_name = 'preflop_fold_TF'
add_constant = False
exog_var_name = ['player', 'outcome_previous_cat']
ordinal_vars = ['slansky', 'seat_map', 'stack_rank']
bool_vars = {}
categorical_drop_vals = {} #{'player': 'Pluribus', 'seat': '6'}  # {'player': 'Bill'}
interaction_vars = [('outcome_previous_loss_TF', 'player'), ('outcome_previous_win_TF', 'player')]   #[('outcome_previous_loss_TF', 'player')]

# filter and map
df_data = df_master.reindex()
df_data['seat_map'] = df_data['seat'].map({'1': '5', '2': '6', '3': '1', '4': '2', '5': '3', '6': '4'})

test = LogisticRegression(endog_name_f=endog_var_name, exog_name_f=exog_var_name, data_f=df_data,
                          add_constant_f=add_constant,
                          convert_ord_list_f=ordinal_vars, convert_bool_dict_f=bool_vars,
                          cat_col_omit_dict_f=categorical_drop_vals, interaction_name_f=interaction_vars)
test.estimate_model()

#--------- SCRATCH CODE ------
df_master.loc[((df_master.outcome > -5000) & (df_master.outcome < -150)) | ((df_master.outcome > 150) & (df_master.outcome < 5000)), 'outcome'].hist()

test.__dict__.keys()

num_min_hands = 10000
g_num_min_hands = None
for g in games.values():
    num_min_hands = min(num_min_hands, len(g.hands))
    g_num_min_hands = g.number if len(g.hands) == num_min_hands else g_num_min_hands

buy_in = 10000
select_g_num = '74'
num_random = 5
import random
for select_g_num in random.sample(list(games.keys()), num_random) + [g_num_min_hands]:
    df_player_stack = pd.DataFrame(columns=list(games[select_g_num].players), index=list(games[select_g_num].hands.keys()))
    for h_num, h in games[select_g_num].hands.items():
        df_player_stack.loc[h_num] = dict([(k, v+buy_in) for k, v in h.start_stack.items()])
    df_player_stack.loc[str(int(df_player_stack.index[-1]) + 1)] = dict([(p, h.start_stack[p] + buy_in + h.outcomes[p]) for p in h.start_stack.keys()])
    df_player_stack.plot(title=('Game %s start stacks.\nEnd amount of money on table: %3.1f' % (select_g_num, df_player_stack.iloc[-1, :].sum()))).set_xlabel('hand_num')

import matplotlib.pyplot as plt

# ----- RESEARCH -----
# QUESTION : within game, what is the proportion of preflop folds following losing hand vs not losing hand? (discount hands where player is blind)
# ----- define research question specific functions -----
# def create_player_df(player_f):
#     def calc_hand_shift_vars(df_ff, shift_var_name_ff):
#         # check to make sure all hands in range are continguous (no missing hands in middle of string)
#         t_df_ff = df_ff[['hand', shift_var_name_ff]].sort_values(by='hand', ascending=True).reindex()
#         if all((df_ff['hand'] - df_ff['hand'].shift(1))[1:] == 1):
#             t_df_ff['prev_' + shift_var_name_ff] = t_df_ff[shift_var_name_ff].shift(1)
#         else:
#             t_df_ff['prev_' + shift_var_name_ff] = None
#         return t_df_ff.drop(columns=[shift_var_name_ff])
#
#     t_records = list()
#     for t_g_num in player_f.game_numbers:
#         g = games[t_g_num]
#         for h_num in range(g.start_hand, g.end_hand):
#             try:
#                 t_records.append({'player': player_f.name, 'game': int(t_g_num), 'hand': h_num,
#                                   'preflop_action': player_f.actions[t_g_num][str(h_num)]['preflop'],
#                                   'outcome': player_f.outcomes[t_g_num][str(h_num)],
#                                   'big_blind': player_f.blinds[t_g_num][str(h_num)]['big'],
#                                   'small_blind': player_f.blinds[t_g_num][str(h_num)]['small'],
#                                   'hole_cards': player_f.cards[t_g_num][str(h_num)],
#                                   'premium_hole': player_f.odds[t_g_num][str(h_num)]['both_hole_premium_cards'],
#                                   'chen_rank': player_f.odds[t_g_num][str(h_num)]['chen'],
#                                   'slansky_rank': player_f.odds[t_g_num][str(h_num)]['slansky'],
#                                   'start_stack': player_f.stacks[t_g_num][str(h_num)],
#                                   'start_stack_rank': player_f.stack_ranks[t_g_num][str(h_num)],
#                                   'seat_numbers': player_f.seat_numbers[t_g_num][str(h_num)]
#                                   })
#             except KeyError:
#                 pass
#
#     df_f = pd.DataFrame(data=t_records).sort_values(by=['game', 'hand'], ascending=True)
#
#     # temp df of shift variables
#     t_df_f = df_f.groupby('game').apply(calc_hand_shift_vars, shift_var_name_ff='outcome').reset_index()
#     t_df_f.drop(columns=['level_1'], inplace=True)
#     df_f = df_f.merge(t_df_f, how='left', on=['game', 'hand'])
#
#     return df_f
#
#
# def create_base_df_wrapper(players_f):
#     df = pd.DataFrame()
#     for p in players_f:
#         df = pd.concat([df, create_player_df(p)], axis=0, ignore_index=True)
#     del p
#
#     # sanity check on games / hands
#     print('\n\n===== Sanity check dataframe =====')
#     print('Total number of games: %d' % df.game.nunique())
#     print('%d players in data set: %s' % (len(df.player.unique()), df.player.unique()))
#     print('Min number of hands per game: %d' % df.groupby('game').hand.nunique().min())
#     print('Max number of hands per game: %d' % df.groupby('game').hand.nunique().max())
#     print('Min number of unique players in a hand across all games: %d' % df.groupby(
#         ['game', 'hand']).player.nunique().min())
#     print('Max number of unique players in a hand across all games: %d' % df.groupby(
#         ['game', 'hand']).player.nunique().max())
#     print('Median number of unique players in a hand across all games: %d' % df.groupby(
#         ['game', 'hand']).player.nunique().median())
#     if not all(df.groupby(['game', 'hand']).apply(
#             lambda x: (x['small_blind'].sum() == 1) and (x['big_blind'].sum() == 1))):
#         print('WARNING: some blinds unaccounted for, check dataframe for missing info')
#
#     # check that stack calculation makes sense
#     df_player_game_stack_check = df.sort_values(['player', 'game', 'hand'], ascending=True).groupby(
#         ['player', 'game']).apply(start_end_stack_comp)
#     if df_player_game_stack_check.sum() != 0:
#         print('WARNING: Check creation of data frame, cumulative stack and outcome calculations do not tie.'
#               '\n\tSee: df_player_game_stack_check_sum')
#
#     # add additional features
#     df['prev_outcome_loss'] = (df.prev_outcome < 0)
#     df['preflop_fold'] = (df['preflop_action'] == 'f')
#     df['relative_start_stack'] = df['start_stack'] / df.groupby(['game', 'hand'])['start_stack'].transform('max')
#     df['rank_start_stack'] = df.groupby(['game', 'hand'])['start_stack'].rank(method='max', ascending=False)
#     df['any_blind'] = (df['small_blind'] | df['big_blind'])
#     df['bot_TF'] = (df['player'] == 'Pluribus')
#
#     if df.groupby(['game', 'hand']).rank_start_stack.max().min() != 6:
#         print('WARNING: stack rankings within game and hand do not always range from 1 to 6. Check data frame.')
#
#     return df
#
#
# def behavior_test(df_f, success_field_name_f, sample_partition_field_name_f):
#     in_sample1_f = df_f[sample_partition_field_name_f]
#     in_sample2_f = ~df_f[sample_partition_field_name_f]
#     success_f = df_f[success_field_name_f]
#     n_sample1 = sum(in_sample1_f)
#     n_sample2 = sum(in_sample2_f)
#     n_sample1_success = sum(in_sample1_f & success_f)
#     n_sample2_success = sum(in_sample2_f & success_f)
#     t_p1, t_p2, t_z, t_p = prop_2samp_ind_large(n1_success=n_sample1_success, n2_success=n_sample2_success,
#                                                 n1=n_sample1, n2=n_sample2)
#     return {'p1': t_p1, 'n1': n_sample1, 'p2': t_p2, 'n2': n_sample2, 'z': t_z, 'pval': t_p}
#
#
# def start_end_stack_comp(df_f):
#     return np.nansum(df_f['start_stack'] + df_f['outcome'] - df_f['start_stack'].shift(-1))

#
# # --- Create dataframe of all player data
# df = create_base_df_wrapper(players)
#
# # --- Conduct hypothesis test
# # Exclude select observations rows
# excl_cond1 = df.small_blind  # less likely to fold if already have money in the pot
# excl_cond2 = df.big_blind  # less likely to fold if already have money in the pot
# excl_cond3 = df.prev_outcome_loss.isnull()  # first hand of game
# excl_cond4 = df.premium_hole  # got lucky with good hole cards no one folds
# use_ind = df.loc[~(excl_cond1 | excl_cond2 | excl_cond3 | excl_cond4)].index
# df_calc = df.loc[use_ind].reindex()
# del use_ind
#
# # Calculate test stats
# sample_partition_binary_field = 'prev_outcome_loss'
# success_event_binary_field = 'preflop_fold'
# t_df = df_calc.groupby('player').apply(behavior_test, success_field_name_f=success_event_binary_field,
#                                        sample_partition_field_name_f=sample_partition_binary_field)
# df_prop_test = pd.DataFrame(list(t_df))
# df_prop_test['player'] = t_df.index
# del t_df
#
# print("Partitioning samples on %s and success event = '%s'." % (
# sample_partition_binary_field, success_event_binary_field))
# print("Test statistic z = (p1 - p2)/stdev and p1 is %s is True" % (sample_partition_binary_field))
# print(df_prop_test)
#
# # --- Logistic regression to incorporate stack size and previous loss
# # df = create_base_df_wrapper(players)    ########
#
# # create training data
# X_col_name = ['any_blind', 'slansky_rank', 'rank_start_stack', 'seat_numbers']     # 'any_blind', 'prev_outcome_loss', 'slansky_rank', 'rank_start_stack', 'bot_TF'; chen_rank, slansky_rank, premium_hole
# y_col_name = ['preflop_fold']
#
# dummy_vars = {'player': 'Pluribus'}  # {'player': 'Pluribus'} base field name: value to drop for identification; if value is None, then no dummies are dropped
#
# # interaction_vars = {}
# # interaction_vars = {'loss_bot': {'var1name': 'prev_outcome_loss', 'var2name': 'bot_TF'}}
# interaction_vars = dict()
# [interaction_vars.update({'loss_' + p: {'var1name': 'prev_outcome_loss', 'var2name': p}}) for p in df.player.unique() if p != 'Pluribus']
#
# add_const = False
#
# df_logistic = df.reindex()
# print('Base regression data set has %d observations and %d columns' % df_logistic.shape)
#
# # process dummy vars
# if len(dummy_vars) > 0:
#     t_df = pd.DataFrame()
#     for t_name, t_excl in dummy_vars.items():
#         t_df_dummies = pd.get_dummies(df_logistic[t_name])
#         t_df = pd.concat([t_df, t_df_dummies], axis=1).drop(columns=t_excl if t_excl is not None else t_df_dummies.columns[0])
#         del t_df_dummies
#     del t_name, t_excl
#     df_logistic = pd.concat([df_logistic, t_df], axis=1)
#     X_col_name = X_col_name + list(t_df.columns)
#     del t_df
#     print('After adding dummies, regression data frame has %d observations and %d variables (incl. dependent)' % df_logistic.shape)
#
# # process interaction terms
# if len(interaction_vars) > 0:
#     t_df = pd.DataFrame()
#     for t_name, t_vars in interaction_vars.items():
#         t_df[t_name] = df_logistic.loc[:, t_vars['var1name']].astype(float) * df_logistic.loc[:, t_vars['var2name']].astype(float)
#     del t_name, t_vars
#     df_logistic = pd.concat([df_logistic, t_df], axis=1)
#     X_col_name = X_col_name + list(t_df.columns)
#     del t_df
#     print('After adding interaction terms regression data frame has %d observations and %d variables (incl. dependent)' % df_logistic.shape)
#
# # drop bad rows for regression
# df_logistic = df_logistic[X_col_name + y_col_name].dropna().reindex()
# print('After dropping rows with null, %d observations and %d columns (incl. dependent) remain' % df_logistic.shape)
#
# if add_const:
#     sm_result = sm.Logit(endog=df_logistic[y_col_name],
#                          exog=sm.add_constant(df_logistic[X_col_name]).astype(float)).fit()
# else:
#     sm_result = sm.Logit(endog=df_logistic[y_col_name], exog=df_logistic[X_col_name].astype(float)).fit()
# print(sm_result.summary2())
#
# # print('Aggregate player specific effects: player dummy plus player * prev hand loss')
# # for p in df.player.unique():
# #     try:
# #         print('%s: %3.1f' % (p, sm_result.params[p] + sm_result.params['loss_' + p]))
# #     except KeyError:
# #         pass
#
#
# # ---- Comparison of traditional player style metrics -----
# def get_select_hands(df_f, true_cond_cols_f):
#     t_index = [True] * df_f.shape[0]
#     # filter dataframe to only include select sub-population based on condition columns
#     for t_col, t_cond in true_cond_cols_f.items():
#         t_index = (t_index & (df_f[t_col] == t_cond))
#     t_df = df_f.loc[t_index]
#
#     # retrieve games and hands for sub-population
#     select_hands_dict_f = dict()
#     for t_g_num in t_df['game'].unique():
#         select_hands_dict_f.update({str(t_g_num): [str(x) for x in t_df.loc[t_df['game'] == t_g_num, 'hand'].unique()]})
#     return select_hands_dict_f
#
#
# looseness_comp = dict()
# for p in players:
#     # excl_cond1 = df.small_blind  # less likely to fold if already have money in the pot
#     # excl_cond2 = df.big_blind  # less likely to fold if already have money in the pot
#     # excl_cond4 = df.premium_hole  # got lucky with good hole cards no one folds
#     t_select_hands_prev_loss = get_select_hands(df.loc[df.player == p.name], true_cond_cols_f={'prev_outcome_loss': True, 'small_blind': False, 'big_blind': False, 'premium_hole': False})
#     t_select_hands_no_prev_loss = get_select_hands(df.loc[df.player == p.name], true_cond_cols_f={'prev_outcome_loss': False, 'small_blind': False, 'big_blind': False, 'premium_hole': False})
#
#     _, n1_success, n1 = p.calc_looseness(t_select_hands_prev_loss)
#     _, n2_success, n2 = p.calc_looseness(t_select_hands_no_prev_loss)
#     t_p1, t_p2, t_z, t_p = prop_2samp_ind_large(n1_success, n2_success, n1, n2)
#     looseness_comp.update({p.name: {'p1': t_p1, 'n1': n1_success, 'p2': t_p2, 'n2': n2_success, 'z': t_z, 'pval': t_p}})
#
#     del p, t_p1, t_p2, t_z, t_p, n1_success, n1, n2_success, n2, t_select_hands_no_prev_loss, t_select_hands_prev_loss
#
# print("Partitioning samples on %s and success event = '%s'." % ('prev_outcome_loss', 'voluntarily played hand (looseness)'))
# print("Test statistic z = (p1 - p2)/stdev and p1 is %s is True" % ('prev_outcome_loss'))
# print(pd.DataFrame.from_dict(looseness_comp, orient='index'))
#
#
# # import pickle
# # with open("python_hand_data.pickle", 'wb') as f:
# #     pickle.dump({'players': players, 'games': games, 'df': df}, f)
