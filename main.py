import os
import numpy as np
import pandas as pd


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
    return p1, p2, z_f


def extract_game_number(fn_f):
    return str(fn_f.split('sample_game_')[1].split('.log')[0])


def meta_game_stats(games_f, print_f=True, long_print_f=False):
    games_with_errors = list()
    games_start_hand_not_zero = dict()
    games_missing_hands = dict()
    games_with_combined_mismatch_players = dict()
    for g_num, g in games_f.items():
        if len(g.error_hands) > 1:
            games_with_errors.append(g_num)
            if long_print_f:
                print("%d hand number errors for game %s" % (g.error_hands, g_num))
        if g.start_hand > 0:
            games_start_hand_not_zero.update({g_num: str(g.start_hand)})
        if len(g.missing_hands) > 0:
            games_missing_hands.update({g_num: g.missing_hands})
        if g.combine_player_diff is not None:
            games_with_combined_mismatch_players.update({g_num: g.combine_player_diff})
            if long_print_f:
                print("%s combined mismatch players for game %s" % (g.combine_player_diff, g_num))

    if print_f:
        print("Number of games processed (game object instantiations): %d" % len(games_f))
        print("Number of games with hand number errors: %d" % len(games_with_errors))
        print("Number of games with starting hand number > 0: %d" % len(games_start_hand_not_zero))
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

    return games_with_errors, games_missing_hands, games_start_hand_not_zero


# ----- Define classes ----
class Hand:
    def __init__(self, raw_hand_string):
        self.hand_data = raw_hand_string
        self.number = self.get_hand_number()
        self.players = self.get_players()
        self.small_blind = self.get_small_blind()
        self.big_blind = self.get_big_blind()
        self.actions = self.get_actions()
        self.outcomes = self.get_outcomes()

    def get_hand_number(self):
        try:
            t_hand_number = self.hand_data.split(':')[1]
            return t_hand_number
        except IndexError:
            return None

    def get_players(self):
        return self.hand_data.split(':')[-1].split('|')

    def get_small_blind(self):
        try:
            return self.players[0]
        except IndexError:
            pass

    def get_big_blind(self):
        try:
            return self.players[1]
        except IndexError:
            pass


    def get_actions(self):
        def get_round_action(round_actors_f, round_actions_f):
            round_dict_f = dict(zip(round_actors_f, [x for x in round_actions_f if x in {'f', 'r', 'c'}]))
            [t_actors.remove(a) for a in [k for k, v in round_dict_f.items() if v == 'f']]
            return round_dict_f
        try:
            t_actions = dict(zip(['preflop', 'flop', 'river', 'turn'], self.hand_data.split(':')[2].split('/')))
            t_actors = self.players[:]

            action_dict_f = {'preflop': get_round_action(round_actors_f=t_actors[2:] + t_actors[0:2],
                                                         round_actions_f=t_actions[
                                                             'preflop'])}  # preflop has different order of betting
            [action_dict_f.update({k: get_round_action(round_actors_f=t_actors, round_actions_f=v)}) for k, v in
             t_actions.items() if k != 'preflop']
            return action_dict_f
        except IndexError:
            return None

    def get_outcomes(self):
        try:
            return dict(zip(self.players, [float(x) for x in self.hand_data.split(':')[4].split('|')]))
        except IndexError:
            return None


class Game:
    def __init__(self, raw_data_string, game_number_f=None):
        self.data = raw_data_string
        self.number = game_number_f
        self.hands = self.parse_hands()
        self.error_hands = None
        self.missing_hands = None
        self.start_hand = None
        self.end_hand = None
        self.players = set()
        self.combine_game_add = None
        self.combine_player_diff = None

        self.summarize_hands()
        self.parse_players()

    def parse_hands(self):
        hands_f = dict()
        for t_hand in [x for x in self.data.split('\n') if x != '']:     # temp = self.data #####
            t_hand_obj = Hand(t_hand)
            hands_f.update({t_hand_obj.number: t_hand_obj})
        return hands_f  # [Hand(x) for x in self.data.split('\n')]

    def get_error_hands(self):
        t_bad_hand_numbers = list()
        for _, t_hand in self.hands.items():
            if t_hand.number is None:
                t_bad_hand_numbers.append(t_hand.number)
        return t_bad_hand_numbers

    def check_for_missing_hand_number(self):
        t_f = [int(x) for x in self.hands.keys() if x is not None]
        t_f.sort()
        return [y - 1 for x, y in zip(t_f, t_f[1:]) if y - x != 1]

    def summarize_hands(self):
        self.error_hands = self.get_error_hands()
        self.missing_hands = self.check_for_missing_hand_number()
        self.start_hand = min([int(x) for x in self.hands.keys() if x is not None])
        self.end_hand = max([int(x) for x in self.hands.keys() if x is not None])

    def parse_players(self):
        for _, x in self.hands.items():
            try:
                self.error_hands.index(x.number)
            except ValueError:
                self.players.update(x.players)

    def combine_games(self, game2, print_f=True):
        self.combine_game_add = game2.number
        combine_player_set_diff_f = self.players - game2.players
        if len(combine_player_set_diff_f) > 0:
            self.combine_player_diff = combine_player_set_diff_f
            if print_f:
                print("WARNING: Different players for combined games %s and %s" % (self.number, game2.number))
                print("Difference: %s" % self.combine_player_diff)
                # print("Players in game %s: %s" % (self.number, self.players))
                # print("Players in game %s: %s" % (game2.number, game2.players))

        self.hands.update(game2.hands)
        self.summarize_hands()

        # re-parse unique players list for game
        self.parse_players()


class Player:
    def __init__(self, name=None):
        self.name = name
        self.game_numbers = None
        self.actions = None
        self.outcomes = None
        self.blinds = None

    def get_game_numbers(self, games_ff):
        return [x for x in games_ff.keys() if self.name in games_ff[x].players]

    def get_game_actions(self, games_ff):
        t_action_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                t_round_dict = dict()
                for t_r, t_a in t_g.hands[str(t_h_num)].actions.items():
                    try:
                        t_round_dict.update({t_r: t_a[self.name]})
                    except KeyError:
                        pass
                t_hand_dict.update({str(t_h_num): t_round_dict})
            t_action_dict.update({t_g_num: t_hand_dict})
        return t_action_dict
    
    def get_game_outcomes(self, games_ff):
        t_outcome_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                t_hand_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].outcomes[self.name]})
                t_outcome_dict.update({t_g_num: t_hand_dict})
        return t_outcome_dict

    def get_blinds(self, games_ff):
        t_blind_dict = dict()
        for t_g_num in self.game_numbers:  # self=p
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                t_hand_dict.update({str(t_h_num): {'big': t_g.hands[str(t_h_num)].big_blind == self.name,
                                                   'small': t_g.hands[str(t_h_num)].small_blind == self.name}})     # self = p
            t_blind_dict.update({t_g_num: t_hand_dict})
        return t_blind_dict

    def add_games_info(self, games_f):
        self.game_numbers = self.get_game_numbers(games_f)
        self.actions = self.get_game_actions(games_f)
        self.outcomes = self.get_game_outcomes(games_f)
        self.blinds = self.get_blinds(games_f)


# ----- process files
print("Number of total file names: %d" % len(fn_data))
games = dict()
for t_fn in fn_data:
    with open(os.path.join(fd_data, t_fn), 'r') as f:
        temp = f.read()
        t_game = Game(temp, game_number_f=extract_game_number(t_fn))
        games.update({t_game.number: t_game})
del t_fn

_, _, _ = meta_game_stats(games)

# ----- combine files with 'b' appended to file name -----
print("\nCombining games with 'b' appended to filename\n")
games_b = [x for x in games.keys() if x.find('b') > -1]
for g1, g2 in zip([x.split('b')[0] for x in games_b], games_b):
    games[g1].combine_games(games[g2], print_f=False)
    games.pop(g2)
del g1, g2

_, _, _ = meta_game_stats(games)

# ----- parse by player -----
all_players = set()
for _, g in games.items():
    all_players = all_players.union(g.players)
del g

players = list()
for p_name in all_players:
    try:
        p = Player(p_name)
        p.add_games_info(games)
        players.append(p)
        del p
    except KeyError:
        pass
del p_name

# ----- RESEARCH -----
# QUESTION : within game, what is the proportion of preflop folds following losing hand vs not losing hand? (discount hands where player is blind)
# --- Create dataframe of all player data
def create_player_df(player_f):
    t_records = list()
    for g_num, g in player_f.actions.items():
        for h_num, h in g.items():
            try:
                t_records.append({'player': player_f.name, 'game': int(g_num), 'hand': int(h_num),
                                  'preflop_fold': h['preflop'] == 'f', 'outcome': player_f.outcomes[g_num][h_num],
                                  'big_blind': player_f.blinds[g_num][h_num]['big'],
                                  'small_blind': player_f.blinds[g_num][h_num]['small']})
            except KeyError:
                pass

    df_f = pd.DataFrame(data=t_records).sort_values(by=['game', 'hand'], ascending=True)
    df_f['prev_outcome'] = df_f.groupby('game')['outcome'].shift(1)
    df_f['prev_outcome_loss'] = df_f.prev_outcome < 0
    return df_f


def behavior_test(df_f, success_field_name_f, denom_field_name_f):
    n_prevloss_preflop_folds = sum((df_f[denom_field_name_f]) & (df_f[success_field_name_f]))
    n_noprevloss_preflop_folds = sum((~df_f[denom_field_name_f]) & (df_f[success_field_name_f]))
    n_prevloss = sum(df_f[denom_field_name_f])
    n_noprevloss = sum(~df_f[denom_field_name_f])
    t_p1, t_p2, t_z = prop_2samp_ind_large(n_prevloss_preflop_folds, n_noprevloss_preflop_folds, n_prevloss, n_noprevloss)
    return {'p1': t_p1, 'n1': n_prevloss_preflop_folds, 'p2': t_p2, 'n2': n_noprevloss_preflop_folds, 'z': t_z}

df = pd.DataFrame()
for p in players:
    df = pd.concat([df, create_player_df(p)], axis=0, ignore_index=True)
del p

# --- Conduct hypothesis test
# Exclude select observations rows
excl_cond1 = df.small_blind
excl_cond2 = df.big_blind
excl_cond3 = df.prev_outcome_loss.isnull()
use_ind = df.loc[~(excl_cond1 | excl_cond2 | excl_cond3)].index
df_calc = df.loc[use_ind].reset_index()

# Calculate test stats
t_df = df_calc.groupby('player').apply(behavior_test, success_field_name_f='preflop_fold', denom_field_name_f='prev_outcome_loss')
df_prop_test = pd.DataFrame(list(t_df))
df_prop_test['player'] = t_df.index
del t_df

print(df_prop_test)
