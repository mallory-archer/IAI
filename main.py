import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from game_classes import *

pd.options.display.max_columns = 25

# ---- Set parameters ----
# fd_data = os.path.join("..", "self", "5H1AI_logs")  # location of data relative to code base path
fd_data = os.path.join("..", "Data", "5H1AI_logs")  # location of data relative to code base path
fn_data = [x for x in os.listdir(fd_data) if x.find('sample_game') > -1]  # list of data file names (log files)


# ---- Define functions -----
def prop_2samp_ind_large(n1_success, n2_success, n1, n2):
    p1 = n1_success / n1
    p2 = n2_success / n2
    phat = (n1_success + n2_success) / (n1 + n2)
    qhat = 1 - phat
    stdev_p1p2_diff = np.sqrt(phat * qhat * ((1 / n1) + (1 / n2)))
    z_f = (p1 - p2) / stdev_p1p2_diff
    p_f = round((1 - norm.cdf(abs(z_f))) * 2, 4)
    return p1, p2, z_f, p_f


def extract_game_number(fn_f):
    return str(fn_f.split('sample_game_')[1].split('.log')[0])


def meta_game_stats(games_f, print_f=True, long_print_f=False):
    games_with_file_parse_errors = list()
    games_with_error_hands = list()
    games_start_hand_not_zero = dict()
    games_missing_hands = dict()
    games_with_combined_mismatch_players = dict()
    min_number_hands_played = (None, 1e6)
    max_number_hands_played = (None, 0)
    for g_num, g in games_f.items():
        if g.hand_parse_errors:
            games_with_file_parse_errors.append(g_num)
        if len(g.error_hands) > 0:
            games_with_error_hands.append(g_num)
            if long_print_f:
                print("%d hands missing information for game %s" % (len(g.error_hands), g_num))
        if g.start_hand > 0:
            games_start_hand_not_zero.update({g_num: str(g.start_hand)})
        if g.total_hands < min_number_hands_played[1]:
            min_number_hands_played = (g_num, g.total_hands)
        if g.total_hands > max_number_hands_played[1]:
            max_number_hands_played = (g_num, g.total_hands)
        if len(g.missing_hands) > 0:
            games_missing_hands.update({g_num: g.missing_hands})
        if g.combine_player_diff is not None:
            games_with_combined_mismatch_players.update({g_num: g.combine_player_diff})
            if long_print_f:
                print("%s combined mismatch players for game %s" % (g.combine_player_diff, g_num))

    if print_f:
        print("Number of games processed (game object instantiations): %d" % len(games_f))
        print("Number of games with hand parse errors: %d" % len(games_with_file_parse_errors))
        print("Number of games with hand calculation errors: %d" % len(games_with_error_hands))
        print("Number of games with starting hand number > 0: %d" % len(games_start_hand_not_zero))
        print("Min %d number of hands in game %s" % (min_number_hands_played[1], min_number_hands_played[0]))
        print("Max %d number of hands in game %s" % (max_number_hands_played[1], max_number_hands_played[0]))
        print("Number of games with hand number missing in sequence: %d" % len(games_missing_hands))
        print("Number of combined games with mismatch players: %d" % len(games_with_combined_mismatch_players))
        # print missing beginning hands
    if long_print_f:
        if len(games_start_hand_not_zero) > 0:
            print("Games with starting hands greater than 0:")
            for g_num, g_start in games_start_hand_not_zero.items():
                print('Game %s starts with hand %s' % (g_num, g_start))
        # print missing hands in sequence
        if len(games_missing_hands) > 0:
            print("Games missing hands:")
            for g_num, g_start in games_missing_hands.items():
                print('Game %s starts with hand %s' % (g_num, g_start))

    return games_with_file_parse_errors, games_with_error_hands, games_missing_hands, games_start_hand_not_zero


# ----- process files
print("Number of total file names: %d" % len(fn_data))
games = dict()
for t_fn in fn_data:
    with open(os.path.join(fd_data, t_fn), 'r') as f:
        temp = f.read()
        t_game = Game(temp, game_number_f=extract_game_number(t_fn))
        games.update({t_game.number: t_game})
del t_fn, t_game

_, _, _, _ = meta_game_stats(games)

# ----- combine files with 'b' appended to file name -----
print("\nCombining games with 'b' appended to filename\n")
games_b = [x for x in games.keys() if x.find('b') > -1]
for g1, g2 in zip([x.split('b')[0] for x in games_b], games_b):
    print('For pair %s and %s, %s missing %s and %s missing %s' % (g1, g2, g1, games[g2].players - games[g1].players, g2, games[g1].players - games[g2].players))
    games[g1].combine_games(games[g2], print_f=False)
    games.pop(g2)
del g1, g2, games_b

_, _, _, _ = meta_game_stats(games)

# ----- drop bad hands -----
print("\nDropping band hands from games\n")
_ = [g.drop_bad_hands() for g in games.values()]

_, _, _, _ = meta_game_stats(games)

# ----- parse by player -----
all_players = set()
for _, g in games.items():
    all_players = all_players.union(g.players)
del g

players = list()
for p_name in all_players:
    p = Player(p_name)
    p.add_games_info(games)
    players.append(p)
    del p
del p_name


# ----- RESEARCH -----
# QUESTION : within game, what is the proportion of preflop folds following losing hand vs not losing hand? (discount hands where player is blind)
# ----- define research question specific functions -----
def create_player_df(player_f):
    def calc_hand_shift_vars(df_ff, shift_var_name_ff):
        # check to make sure all hands in range are continguous (no missing hands in middle of string)
        t_df_ff = df_ff[['hand', shift_var_name_ff]].sort_values(by='hand', ascending=True).reindex()
        if all((df_ff['hand'] - df_ff['hand'].shift(1))[1:] == 1):
            t_df_ff['prev_' + shift_var_name_ff] = t_df_ff[shift_var_name_ff].shift(1)
        else:
            t_df_ff['prev_' + shift_var_name_ff] = None
        return t_df_ff.drop(columns=[shift_var_name_ff])

    t_records = list()
    for t_g_num in player_f.game_numbers:
        g = games[t_g_num]
        for h_num in range(g.start_hand, g.end_hand):
            try:
                t_records.append({'player': player_f.name, 'game': int(t_g_num), 'hand': h_num,
                                  'preflop_action': player_f.actions[t_g_num][str(h_num)]['preflop'],
                                  'outcome': player_f.outcomes[t_g_num][str(h_num)],
                                  'big_blind': player_f.blinds[t_g_num][str(h_num)]['big'],
                                  'small_blind': player_f.blinds[t_g_num][str(h_num)]['small'],
                                  'hole_cards': player_f.cards[t_g_num][str(h_num)],
                                  'premium_hole': player_f.odds[t_g_num][str(h_num)]['both_hole_premium_cards'],
                                  'chen_rank': player_f.odds[t_g_num][str(h_num)]['chen'],
                                  'slansky_rank': player_f.odds[t_g_num][str(h_num)]['slansky'],
                                  'start_stack': player_f.stacks[t_g_num][str(h_num)],
                                  'start_stack_rank': player_f.stack_ranks[t_g_num][str(h_num)],
                                  'seat_numbers': player_f.seat_numbers[t_g_num][str(h_num)]
                                  })
            except KeyError:
                pass

    df_f = pd.DataFrame(data=t_records).sort_values(by=['game', 'hand'], ascending=True)

    # temp df of shift variables
    t_df_f = df_f.groupby('game').apply(calc_hand_shift_vars, shift_var_name_ff='outcome').reset_index()
    t_df_f.drop(columns=['level_1'], inplace=True)
    df_f = df_f.merge(t_df_f, how='left', on=['game', 'hand'])

    return df_f


def create_base_df_wrapper(players_f):
    df = pd.DataFrame()
    for p in players_f:
        df = pd.concat([df, create_player_df(p)], axis=0, ignore_index=True)
    del p

    # sanity check on games / hands
    print('\n\n===== Sanity check dataframe =====')
    print('Total number of games: %d' % df.game.nunique())
    print('%d players in data set: %s' % (len(df.player.unique()), df.player.unique()))
    print('Min number of hands per game: %d' % df.groupby('game').hand.nunique().min())
    print('Max number of hands per game: %d' % df.groupby('game').hand.nunique().max())
    print('Min number of unique players in a hand across all games: %d' % df.groupby(
        ['game', 'hand']).player.nunique().min())
    print('Max number of unique players in a hand across all games: %d' % df.groupby(
        ['game', 'hand']).player.nunique().max())
    print('Median number of unique players in a hand across all games: %d' % df.groupby(
        ['game', 'hand']).player.nunique().median())
    if not all(df.groupby(['game', 'hand']).apply(
            lambda x: (x['small_blind'].sum() == 1) and (x['big_blind'].sum() == 1))):
        print('WARNING: some blinds unaccounted for, check dataframe for missing info')

    # check that stack calculation makes sense
    df_player_game_stack_check = df.sort_values(['player', 'game', 'hand'], ascending=True).groupby(
        ['player', 'game']).apply(start_end_stack_comp)
    if df_player_game_stack_check.sum() != 0:
        print('WARNING: Check creation of data frame, cumulative stack and outcome calculations do not tie.'
              '\n\tSee: df_player_game_stack_check_sum')

    # add additional features
    df['prev_outcome_loss'] = (df.prev_outcome < 0)
    df['preflop_fold'] = (df['preflop_action'] == 'f')
    df['relative_start_stack'] = df['start_stack'] / df.groupby(['game', 'hand'])['start_stack'].transform('max')
    df['rank_start_stack'] = df.groupby(['game', 'hand'])['start_stack'].rank(method='max', ascending=False)
    df['any_blind'] = (df['small_blind'] | df['big_blind'])
    df['bot_TF'] = (df['player'] == 'Pluribus')

    if df.groupby(['game', 'hand']).rank_start_stack.max().min() != 6:
        print('WARNING: stack rankings within game and hand do not always range from 1 to 6. Check data frame.')

    return df


def behavior_test(df_f, success_field_name_f, sample_partition_field_name_f):
    in_sample1_f = df_f[sample_partition_field_name_f]
    in_sample2_f = ~df_f[sample_partition_field_name_f]
    success_f = df_f[success_field_name_f]
    n_sample1 = sum(in_sample1_f)
    n_sample2 = sum(in_sample2_f)
    n_sample1_success = sum(in_sample1_f & success_f)
    n_sample2_success = sum(in_sample2_f & success_f)
    t_p1, t_p2, t_z, t_p = prop_2samp_ind_large(n1_success=n_sample1_success, n2_success=n_sample2_success,
                                                n1=n_sample1, n2=n_sample2)
    return {'p1': t_p1, 'n1': n_sample1, 'p2': t_p2, 'n2': n_sample2, 'z': t_z, 'pval': t_p}


def start_end_stack_comp(df_f):
    return np.nansum(df_f['start_stack'] + df_f['outcome'] - df_f['start_stack'].shift(-1))


# --- Create dataframe of all player data
df = create_base_df_wrapper(players)

# --- Conduct hypothesis test
# Exclude select observations rows
excl_cond1 = df.small_blind  # less likely to fold if already have money in the pot
excl_cond2 = df.big_blind  # less likely to fold if already have money in the pot
excl_cond3 = df.prev_outcome_loss.isnull()  # first hand of game
excl_cond4 = df.premium_hole  # got lucky with good hole cards no one folds
use_ind = df.loc[~(excl_cond1 | excl_cond2 | excl_cond3 | excl_cond4)].index
df_calc = df.loc[use_ind].reindex()
del use_ind

# Calculate test stats
sample_partition_binary_field = 'prev_outcome_loss'
success_event_binary_field = 'preflop_fold'
t_df = df_calc.groupby('player').apply(behavior_test, success_field_name_f=success_event_binary_field,
                                       sample_partition_field_name_f=sample_partition_binary_field)
df_prop_test = pd.DataFrame(list(t_df))
df_prop_test['player'] = t_df.index
del t_df

print("Partitioning samples on %s and success event = '%s'." % (
sample_partition_binary_field, success_event_binary_field))
print("Test statistic z = (p1 - p2)/stdev and p1 is %s is True" % (sample_partition_binary_field))
print(df_prop_test)

# --- Logistic regression to incorporate stack size and previous loss
# df = create_base_df_wrapper(players)    ########

# create training data
X_col_name = ['any_blind', 'slansky_rank', 'rank_start_stack', 'seat_numbers']     # 'any_blind', 'prev_outcome_loss', 'slansky_rank', 'rank_start_stack', 'bot_TF'; chen_rank, slansky_rank, premium_hole
y_col_name = ['preflop_fold']

dummy_vars = {'player': 'Pluribus'}  # {'player': 'Pluribus'} base field name: value to drop for identification; if value is None, then no dummies are dropped

# interaction_vars = {}
# interaction_vars = {'loss_bot': {'var1name': 'prev_outcome_loss', 'var2name': 'bot_TF'}}
interaction_vars = dict()
[interaction_vars.update({'loss_' + p: {'var1name': 'prev_outcome_loss', 'var2name': p}}) for p in df.player.unique() if p != 'Pluribus']

add_const = False

df_logistic = df.reindex()
print('Base regression data set has %d observations and %d columns' % df_logistic.shape)

# process dummy vars
if len(dummy_vars) > 0:
    t_df = pd.DataFrame()
    for t_name, t_excl in dummy_vars.items():
        t_df_dummies = pd.get_dummies(df_logistic[t_name])
        t_df = pd.concat([t_df, t_df_dummies], axis=1).drop(columns=t_excl if t_excl is not None else t_df_dummies.columns[0])
        del t_df_dummies
    del t_name, t_excl
    df_logistic = pd.concat([df_logistic, t_df], axis=1)
    X_col_name = X_col_name + list(t_df.columns)
    del t_df
    print('After adding dummies, regression data frame has %d observations and %d variables (incl. dependent)' % df_logistic.shape)

# process interaction terms
if len(interaction_vars) > 0:
    t_df = pd.DataFrame()
    for t_name, t_vars in interaction_vars.items():
        t_df[t_name] = df_logistic.loc[:, t_vars['var1name']].astype(float) * df_logistic.loc[:, t_vars['var2name']].astype(float)
    del t_name, t_vars
    df_logistic = pd.concat([df_logistic, t_df], axis=1)
    X_col_name = X_col_name + list(t_df.columns)
    del t_df
    print('After adding interaction terms regression data frame has %d observations and %d variables (incl. dependent)' % df_logistic.shape)

# drop bad rows for regression
df_logistic = df_logistic[X_col_name + y_col_name].dropna().reindex()
print('After dropping rows with null, %d observations and %d columns (incl. dependent) remain' % df_logistic.shape)

if add_const:
    sm_result = sm.Logit(endog=df_logistic[y_col_name],
                         exog=sm.add_constant(df_logistic[X_col_name]).astype(float)).fit()
else:
    sm_result = sm.Logit(endog=df_logistic[y_col_name], exog=df_logistic[X_col_name].astype(float)).fit()
print(sm_result.summary2())

# print('Aggregate player specific effects: player dummy plus player * prev hand loss')
# for p in df.player.unique():
#     try:
#         print('%s: %3.1f' % (p, sm_result.params[p] + sm_result.params['loss_' + p]))
#     except KeyError:
#         pass


# ---- Comparison of traditional player style metrics -----
def get_select_hands(df_f, true_cond_cols_f):
    t_index = [True] * df_f.shape[0]
    # filter dataframe to only include select sub-population based on condition columns
    for t_col, t_cond in true_cond_cols_f.items():
        t_index = (t_index & (df_f[t_col] == t_cond))
    t_df = df_f.loc[t_index]

    # retrieve games and hands for sub-population
    select_hands_dict_f = dict()
    for t_g_num in t_df['game'].unique():
        select_hands_dict_f.update({str(t_g_num): [str(x) for x in t_df.loc[t_df['game'] == t_g_num, 'hand'].unique()]})
    return select_hands_dict_f


looseness_comp = dict()
for p in players:
    # excl_cond1 = df.small_blind  # less likely to fold if already have money in the pot
    # excl_cond2 = df.big_blind  # less likely to fold if already have money in the pot
    # excl_cond4 = df.premium_hole  # got lucky with good hole cards no one folds
    t_select_hands_prev_loss = get_select_hands(df.loc[df.player == p.name], true_cond_cols_f={'prev_outcome_loss': True, 'small_blind': False, 'big_blind': False, 'premium_hole': False})
    t_select_hands_no_prev_loss = get_select_hands(df.loc[df.player == p.name], true_cond_cols_f={'prev_outcome_loss': False, 'small_blind': False, 'big_blind': False, 'premium_hole': False})

    _, n1_success, n1 = p.calc_looseness(t_select_hands_prev_loss)
    _, n2_success, n2 = p.calc_looseness(t_select_hands_no_prev_loss)
    t_p1, t_p2, t_z, t_p = prop_2samp_ind_large(n1_success, n2_success, n1, n2)
    looseness_comp.update({p.name: {'p1': t_p1, 'n1': n1_success, 'p2': t_p2, 'n2': n2_success, 'z': t_z, 'pval': t_p}})

    del p, t_p1, t_p2, t_z, t_p, n1_success, n1, n2_success, n2, t_select_hands_no_prev_loss, t_select_hands_prev_loss

print("Partitioning samples on %s and success event = '%s'." % ('prev_outcome_loss', 'voluntarily played hand (looseness)'))
print("Test statistic z = (p1 - p2)/stdev and p1 is %s is True" % ('prev_outcome_loss'))
print(pd.DataFrame.from_dict(looseness_comp, orient='index'))

# import pickle
# import pickle
# with open("python_hand_data.pickle", 'wb') as f:
#     pickle.dump({'players': players, 'games': games, 'df': df}, f)
