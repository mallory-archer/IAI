import os

# ---- Set parameters ----
fd_data = os.path.join("..", "Data", "5H1AI_logs")  # location of data relative to code base path
fn_data = [x for x in os.listdir(fd_data) if x.find('sample_game') > -1]     # list of data file names (log files)


# ---- Define functions -----
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

    def get_hand_number(self):
        try:
            t_hand_number = self.hand_data.split(':')[1]
            return t_hand_number
        except IndexError:
            return None

    def get_players(self):
        return self.hand_data.split(':')[-1].split('|')


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
        for t_hand in [x for x in self.data.split('\n') if x != '']:
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


# ----- process files
print("Number of total file names: %d" % len(fn_data))
games = dict()
for t_fn in fn_data:
    with open(os.path.join(fd_data, t_fn), 'r') as f:
        temp = f.read()
        t_game = Game(temp, game_number_f=extract_game_number(t_fn))
        games.update({t_game.number: t_game})

_, _, _ = meta_game_stats(games)

# ----- combine files with 'b' appended to file name -----
print("\nCombining games with 'b' appended to filename\n")
games_b = [x for x in games.keys() if x.find('b') > -1]
for g1, g2 in zip([x.split('b')[0] for x in games_b], games_b):
    games[g1].combine_games(games[g2], print_f=False)
    games.pop(g2)

_, _, _ = meta_game_stats(games)

