import os
import numpy as np
import pandas as pd
import json
from scipy.stats import norm
import statsmodels.api as sm

pd.options.display.max_columns = 25

# ---- Set parameters ----
fd_data = os.path.join("..", "Data", "5H1AI_logs")  # location of data relative to code base path
fn_data = [x for x in os.listdir(fd_data) if x.find('sample_game') > -1]     # list of data file names (log files)


# ---- Define functions -----
def prop_2samp_ind_large(n1_success, n2_success, n1, n2):
    p1 = n1_success / n1
    p2 = n2_success / n2
    phat = (n1_success + n2_success) / (n1 + n2)
    qhat = 1 - phat
    stdev_p1p2_diff = np.sqrt(phat * qhat * ((1 / n1) + (1 / n2)))
    z_f = (p1 - p2) / stdev_p1p2_diff
    p_f = round((1 - norm.cdf(abs(z_f)))*2, 4)
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


# ----- Define classes ----
# class Hand:
#     def __init__(self, raw_hand_string):
#         self.hand_data = raw_hand_string
#         self.number = self.get_hand_number()
#         self.players = self.get_players()
#         self.small_blind = self.get_small_blind()
#         self.big_blind = self.get_big_blind()
#         self.cards = self.get_cards()
#         self.odds = self.get_odds()
#         self.actions = self.get_actions()
#         self.outcomes = self.get_outcomes()
#         self.missing_fields = list()
#         self.start_stack = None     # calculated in Game object because it is based on change log of previous hands
#
#         # check for initialization of select attributes
#         self.check_player_completeness()
#
#     def get_hand_number(self):
#         try:
#             t_hand_number = self.hand_data.split(':')[1]
#             return t_hand_number
#         except IndexError:
#             return None
#
#     def get_players(self):
#         return [x.rstrip() for x in self.hand_data.split(':')[-1].split('|')]
#
#     def get_small_blind(self):
#         try:
#             return self.players[0]
#         except IndexError:
#             pass
#
#     def get_big_blind(self):
#         try:
#             return self.players[1]
#         except IndexError:
#             pass
#
#     def get_cards(self):
#         try:
#             t_all_cards = self.hand_data.split(':')[3].split('|')
#             if len(t_all_cards) > len(self.players):
#                 t_board_cards = t_all_cards[-1].split('/')
#                 t_hole_cards = t_all_cards[:-1] + [t_board_cards.pop(0)]  # last set of hole cards splits to board because of "/" "|" convention
#                 return {'hole_cards': dict(zip(self.players, t_hole_cards)), 'board_cards': t_board_cards}
#             else:
#                 t_hole_cards = t_all_cards
#                 return {'hole_cards': dict(zip(self.players, t_hole_cards))}
#         except IndexError:
#             return None
#
#     def get_odds(self):
#         # simplistic proxy - could be refined
#         try:
#             t_cards = self.cards['hole_cards']
#             premium_cards_f = {'A', 'K', 'Q', 'J'}
#             odds_dict_f = dict()
#             for k, v in t_cards.items():
#                 odds_dict_f.update({k: {'both_hole_premium_cards': (v[0] in premium_cards_f) & (v[2] in premium_cards_f)}})
#             return odds_dict_f
#         except TypeError:
#             return None
#
#     def get_actions(self):
#         def get_round_action(round_actors_f, round_actions_f):
#             round_dict_f = dict(zip(round_actors_f, [x for x in round_actions_f if x in {'f', 'r', 'c'}]))
#             [t_actors.remove(a) for a in [k for k, v in round_dict_f.items() if v == 'f']]
#             return round_dict_f
#         try:
#             t_actions = dict(zip(['preflop', 'flop', 'river', 'turn'], self.hand_data.split(':')[2].split('/')))
#             t_actors = self.players[:]
#
#             # adjust preflop actions to account for all folds defaulting to big blind gets pot; label as "call" for big blind
#             if (len(t_actions['preflop']) < len(t_actors)) and (all([x=='f' for x in t_actions['preflop']])):
#                 t_actions['preflop'] += 'c'
#
#             action_dict_f = {'preflop': get_round_action(round_actors_f=t_actors[2:] + t_actors[0:2], round_actions_f=t_actions['preflop'])}  # preflop has different order of betting
#             [action_dict_f.update({k: get_round_action(round_actors_f=t_actors, round_actions_f=v)}) for k, v in t_actions.items() if k != 'preflop']
#             return action_dict_f
#         except IndexError:
#             return None
#
#     def get_outcomes(self):
#         try:
#             return dict(zip(self.players, [float(x) for x in self.hand_data.split(':')[4].split('|')]))
#         except IndexError:
#             return None
#
#     def check_player_completeness(self, check_atts_ff=None):
#         try:
#             if check_atts_ff is None:
#                 check_atts_ff = {'cards': ['hole_cards'], 'odds': [], 'actions': ['preflop'], 'outcomes': []}
#             for t_att, t_value in check_atts_ff.items():
#                 if len(t_value) > 0:
#                     for t_key in t_value:
#                         t_diff = set(self.players).difference(getattr(self, t_att)[t_key].keys())
#                         if len(t_diff) > 0:
#                             self.missing_fields.append((t_att, (t_key, t_diff)))
#                 else:
#                     t_diff = set(self.players).difference(getattr(self, t_att).keys())
#                     if len(t_diff) > 0:
#                         self.missing_fields.append((t_att, t_diff))
#         except TypeError:
#             pass
#
#     def print(self):
#         print(json.dumps(self.__dict__, indent=4))
#
#
# class Game:
#     def __init__(self, raw_data_string, game_number_f=None):
#         self.data = raw_data_string
#         self.number = game_number_f
#         self.hands = self.parse_hands()
#
#         self.players = set()
#         self.hand_parse_errors = False
#         self.error_hands = None
#         self.missing_hands = None
#         self.start_hand = None
#         self.end_hand = None
#         self.total_hands = None
#         self.final_outcome = None
#
#         self.combine_game_add = None
#         self.combine_player_diff = None
#
#         self.summarize_hands()
#
#     def parse_hands(self):
#         hands_f = dict()
#         for t_hand in [x for x in self.data.split('\n') if x != '']:
#             t_hand_obj = Hand(t_hand)
#             hands_f.update({t_hand_obj.number: t_hand_obj})
#         return hands_f  # [Hand(x) for x in self.data.split('\n')]
#
#     def get_error_hands(self):
#         t_hand_numbers_missing_fields = list()
#         self.hand_parse_errors = False  ##### This resets hand_parse_errors, if this flag covers anything besides None key, this reset will lose that info as currently written
#         for _, t_hand in self.hands.items():
#             if t_hand.number is None:
#                 self.hand_parse_errors = True
#             if len(t_hand.missing_fields) > 0:
#                 t_hand_numbers_missing_fields.append(t_hand.number)
#         return t_hand_numbers_missing_fields
#
#     def check_for_missing_hand_number(self):
#         t_f = [int(x) for x in self.hands.keys() if x is not None]
#         t_f.sort()
#         return [y - 1 for x, y in zip(t_f, t_f[1:]) if y - x != 1]
#
#     def parse_players(self):
#         for _, x in self.hands.items():
#             if x.number is not None:
#                 self.players.update(x.players)
#
#     def get_stack_sizes(self):
#         if len(self.check_for_missing_hand_number()) > 0:
#             print('ERROR:cannot calculate stack size, game missing consecutively numbered hands.')
#         else:
#             self.hands[str(self.start_hand)].start_stack = dict(zip(self.players, [0] * len(self.players)))  # stack at beginning of game (all players set at 0)
#             for t_h_num in range(int(self.start_hand) + 1, int(self.end_hand) + 1):
#                 self.hands[str(t_h_num)].start_stack = self.hands[str(t_h_num - 1)].start_stack.copy()  # initialize stack dictionary for hand
#
#                 # Check to make sure all player outcomes are accounted for
#                 if sum(self.hands[str(t_h_num - 1)].outcomes.values()) != 0:
#                     print('WARNING: game %s hand %d is not a zero sum outcome hand' % (self.number, t_h_num))
#
#                 for t_p, t_s in self.hands[str(t_h_num - 1)].outcomes.items():
#                     try:
#                         self.hands[str(t_h_num)].start_stack[t_p] = t_s + self.hands[str(t_h_num - 1)].start_stack[t_p]  # add stack at beginning of previous hand + outcome of previous hand
#                     except KeyError:
#                         pass
#
#             # add total game outcome to game object
#             self.final_outcome = self.hands[str(self.end_hand)].start_stack.copy()
#             for t_p, t_s in self.hands[str(self.end_hand)].outcomes.items():
#                 self.final_outcome.update({t_p: self.final_outcome[t_p] + t_s})
#             if sum(self.final_outcome.values()) != 0:
#                 print('WARNING: Final outcome of game %s is not zero-sum over all players, %f unaccounted for' % (self.number, sum(self.final_outcome.values())))
#
#         return None
#
#     def summarize_hands(self):
#         self.parse_players()
#         self.missing_hands = self.check_for_missing_hand_number()
#         self.start_hand = min([int(x) for x in self.hands.keys() if x is not None])
#         self.end_hand = max([int(x) for x in self.hands.keys() if x is not None])
#         self.total_hands = len(self.hands.keys())
#         self.get_stack_sizes()
#         self.error_hands = self.get_error_hands()
#
#     def combine_games(self, game2, print_f=True):
#         self.combine_game_add = game2.number
#         combine_player_set_diff_f = self.players - game2.players
#         if len(combine_player_set_diff_f) > 0:
#             self.combine_player_diff = combine_player_set_diff_f
#             if print_f:
#                 print("WARNING: Different players for combined games %s and %s" % (self.number, game2.number))
#                 print("Difference: %s" % self.combine_player_diff)
#
#         self.hands.update(game2.hands)
#         self.summarize_hands()
#
#     def drop_bad_hands(self, hand_num_null_TF=True):
#         t_num_hands_dropped = 0
#         t_hand_numbers = list(self.hands.keys())    # structured as such so that dictionary doesn't change size during iteration
#         for t_h_num in t_hand_numbers:
#             t_pop = False
#             if hand_num_null_TF:
#                 if t_h_num is None:
#                     t_pop = True
#             # can add more conditions for dropping hand here in same format as hand_num_null_TF
#
#             if t_pop:
#                 self.hands.pop(t_h_num)
#                 t_num_hands_dropped += 1
#
#         # summarize output
#         if t_num_hands_dropped > 0:
#             print("Dropped %d bad hands for game %s" % (t_num_hands_dropped, self.number))
#         self.summarize_hands()
#
#     def print(self):
#         for t_key, t_value in self.__dict__.items():
#             if t_key != 'hands':
#                 if t_key != 'players':
#                     print(json.dumps({t_key: t_value}, indent=4))
#                 else:
#                     print(json.dumps({t_key: list(t_value)}, indent=4))
#
#
# class Player:
#     def __init__(self, name=None):
#         self.name = name
#         self.game_numbers = None
#         self.actions = None
#         self.outcomes = None
#         self.blinds = None
#         self.cards = None
#         self.odds = None
#         self.stacks = None
#         self.looseness = None
#
#     def get_game_numbers(self, games_ff):
#         return [x for x in games_ff.keys() if self.name in games_ff[x].players]
#
#     def get_game_cards(self, games_ff):
#         t_card_dict = dict()
#         for t_g_num in self.game_numbers:
#             t_g = games_ff[t_g_num]
#             t_hand_dict = dict()
#             for t_h_num in range(t_g.start_hand, t_g.end_hand):
#                 try:
#                     t_hand_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].cards['hole_cards'][self.name]})
#                 except KeyError:
#                     pass
#             t_card_dict.update({t_g_num: t_hand_dict})
#         return t_card_dict
#
#     def get_game_odds(self, games_ff):
#         t_odds_dict = dict()
#         for t_g_num in self.game_numbers:
#             t_g = games_ff[t_g_num]
#             t_hand_dict = dict()
#             for t_h_num in range(t_g.start_hand, t_g.end_hand):
#                 try:
#                     t_hand_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].odds[self.name]['both_hole_premium_cards']})
#                 except KeyError:
#                     pass
#             t_odds_dict.update({t_g_num: t_hand_dict})
#         return t_odds_dict
#
#     def get_game_actions(self, games_ff):
#         t_action_dict = dict()
#         for t_g_num in self.game_numbers:
#             t_g = games_ff[t_g_num]
#             t_hand_dict = dict()
#             for t_h_num in range(t_g.start_hand, t_g.end_hand):
#                 t_round_dict = dict()
#                 for t_r, t_a in t_g.hands[str(t_h_num)].actions.items():
#                     try:
#                         t_round_dict.update({t_r: t_a[self.name]})
#                     except KeyError:
#                         pass
#                 t_hand_dict.update({str(t_h_num): t_round_dict})
#             t_action_dict.update({t_g_num: t_hand_dict})
#         return t_action_dict
#
#     def get_game_outcomes(self, games_ff):
#         t_outcome_dict = dict()
#         for t_g_num in self.game_numbers:
#             t_g = games_ff[t_g_num]
#             t_hand_dict = dict()
#             for t_h_num in range(t_g.start_hand, t_g.end_hand):
#                 try:
#                     t_hand_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].outcomes[self.name]})
#                 except KeyError:
#                     pass
#             t_outcome_dict.update({t_g_num: t_hand_dict})
#         return t_outcome_dict
#
#     def get_blinds(self, games_ff):
#         t_blind_dict = dict()
#         for t_g_num in self.game_numbers:  # self=p
#             t_g = games_ff[t_g_num]
#             t_hand_dict = dict()
#             for t_h_num in range(t_g.start_hand, t_g.end_hand):
#                 t_hand_dict.update({str(t_h_num): {'big': t_g.hands[str(t_h_num)].big_blind == self.name,
#                                                    'small': t_g.hands[str(t_h_num)].small_blind == self.name}})     # self = p
#             t_blind_dict.update({t_g_num: t_hand_dict})
#         return t_blind_dict
#
#     def get_stacks(self, games_ff):
#         t_stack_dict = dict()
#         for t_g_num in self.game_numbers:
#             t_g = games_ff[t_g_num]
#             t_hand_dict = dict()
#             for t_h_num in range(t_g.start_hand, t_g.end_hand):
#                 t_hand_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].start_stack[self.name]})
#             t_stack_dict.update({t_g_num: t_hand_dict})
#         return t_stack_dict
#
#     def calc_looseness(self, select_hands_ff=None):
#         # if no subset is prescribed for calculation use all available player data
#         if select_hands_ff is None:
#             select_hands_ff = dict()
#             for t_g_num in self.game_numbers:
#                 select_hands_ff.update({t_g_num: list(self.actions[t_g_num].keys())})
#
#         t_hand_count = 0
#         t_voluntary_play_count = 0
#         for t_g_num, t_h_nums in select_hands_ff.items():
#             for t_h_num in t_h_nums:
#                 t_hand_count += int((not any(self.blinds[t_g_num][t_h_num].values())))
#                 try:    # if player is eliminated actions dictionary is empty
#                     t_voluntary_play_count += int((not any(self.blinds[t_g_num][t_h_num].values())) and
#                                                   (self.actions[t_g_num][t_h_num]['preflop'] == 'r' or self.actions[t_g_num][t_h_num]['preflop'] == 'c'))   # count hand if not blind and raised or called, #### pre-flop only configured
#                 except KeyError:
#                     pass
#         return t_voluntary_play_count / t_hand_count, t_voluntary_play_count, t_hand_count
#
#     def add_games_info(self, games_f):
#         self.game_numbers = self.get_game_numbers(games_f)
#         self.actions = self.get_game_actions(games_f)
#         self.outcomes = self.get_game_outcomes(games_f)
#         self.blinds = self.get_blinds(games_f)
#         self.cards = self.get_game_cards(games_f)
#         self.odds = self.get_game_odds(games_f)
#         self.stacks = self.get_stacks(games_f)
#         self.looseness, _, _ = self.calc_looseness()
#
#     def print(self):
#         print(json.dumps(self.__dict__, indent=4))


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
                                  'premium_hole': player_f.odds[t_g_num][str(h_num)],
                                  'start_stack': player_f.stacks[t_g_num][str(h_num)]
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
    df['prev_outcome_loss'] = df.prev_outcome < 0
    df['preflop_fold'] = df['preflop_action'] == 'f'
    df['relative_start_stack'] = df['start_stack'] / df.groupby(['game', 'hand'])['start_stack'].transform('max')
    df['rank_start_stack'] = df.groupby(['game', 'hand'])['start_stack'].rank(method='max', ascending=False)
    df['any_blind'] = df['small_blind'] | df['big_blind']

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
    t_p1, t_p2, t_z, t_p = prop_2samp_ind_large(n1_success=n_sample1_success, n2_success=n_sample2_success, n1=n_sample1, n2=n_sample2)
    return {'p1': t_p1, 'n1': n_sample1, 'p2': t_p2, 'n2': n_sample2, 'z': t_z, 'pval': t_p}


def start_end_stack_comp(df_f):
    return np.nansum(df_f['start_stack'] + df_f['outcome'] - df_f['start_stack'].shift(-1))


# --- Create dataframe of all player data
df = create_base_df_wrapper(players)

# --- Conduct hypothesis test
# Exclude select observations rows
excl_cond1 = df.small_blind     # less likely to fold if already have money in the pot
excl_cond2 = df.big_blind       # less likely to fold if already have money in the pot
excl_cond3 = df.prev_outcome_loss.isnull()  # first hand of game
excl_cond4 = df.premium_hole    # got lucky with good hole cards no one folds
use_ind = df.loc[~(excl_cond1 | excl_cond2 | excl_cond3 | excl_cond4)].index
df_calc = df.loc[use_ind].reindex()
del use_ind

# Calculate test stats
sample_partition_binary_field = 'prev_outcome_loss'
success_event_binary_field = 'preflop_fold'
t_df = df_calc.groupby('player').apply(behavior_test, success_field_name_f=success_event_binary_field, sample_partition_field_name_f=sample_partition_binary_field)
df_prop_test = pd.DataFrame(list(t_df))
df_prop_test['player'] = t_df.index
del t_df

print("Partitioning samples on %s and success event = '%s'." % (sample_partition_binary_field, success_event_binary_field))
print("Test statistic z = (p1 - p2)/stdev and p1 is %s is True" % (sample_partition_binary_field))
print(df_prop_test)

# --- Logistic regression to incorporate stack size and previous loss
# df = create_base_df_wrapper(players)    ########

# create training data
X_col_name = ['any_blind', 'prev_outcome_loss', 'premium_hole', 'rank_start_stack']
y_col_name = ['preflop_fold']
dummy_vars = {'player': 'Pluribus'}     # base field name: value to drop for identification; if value is None, then no dummies are dropped
add_const = False

# process dummy vars
if len(dummy_vars) > 0:
    df_logistic = df[X_col_name + list(dummy_vars.keys()) + y_col_name].dropna().reindex()
    for t_name, t_excl in dummy_vars.items():
        t_df_dummies = pd.get_dummies(df_logistic[t_name])
        if t_excl is not None:
            t_cols_to_drop = [t_name, t_excl]
        else:
            t_cols_to_drop = [t_name]
        del t_excl, t_name
        df_logistic = pd.concat([df_logistic, t_df_dummies], axis=1).drop(columns=t_cols_to_drop)
        X_col_name = [x for x in df_logistic.columns if x != y_col_name[0]]
        del t_cols_to_drop, t_df_dummies
else:
    df_logistic = df[X_col_name + y_col_name].dropna().reindex()
print('Regression data frame has %d observations and %d variables (incl. dependent)' %df_logistic.shape)

if add_const:
    sm_result = sm.Logit(endog=df_logistic[y_col_name], exog=sm.add_constant(df_logistic[X_col_name]).astype(float)).fit()
else:
    sm_result = sm.Logit(endog=df_logistic[y_col_name], exog=df_logistic[X_col_name].astype(float)).fit()
print(sm_result.summary2())


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
# with open("python_hand_data.pickle", 'wb') as f:
#     pickle.dump({'players': players, 'games': games, 'df': df}, f)
