import os

# ---- Set parameters ----
fd_data = os.path.join("..", "Data", "5H1AI_logs")  # location of data relative to code base path
fn_data = [x for x in os.listdir(fd_data) if x.find('sample_game') > -1]     # list of data file names (log files)


# ---- Define functions -----
def extract_game_number(fn_f):
    return str(fn_f.split('sample_game_')[1].split('.log')[0])


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
        self.error_hands = self.count_hand_number_errors()
        self.players = set()

        self.parse_players()

    def parse_hands(self):
        return [Hand(x) for x in self.data.split('\n')]

    def count_hand_number_errors(self):
        t_count = 0
        for t_hand in self.hands:
            if t_hand.number is None:
                t_count += 1
        return t_count

    def parse_players(self):
        for x in self.hands:
            self.players.update(x.players)


# ----- process files
games = list()
for t_fn in fn_data:
    with open(os.path.join(fd_data, t_fn), 'r') as f:
        games.append(Game(f.read(), game_number_f=extract_game_number(t_fn)))

print("Number of total file names: %d" % len(fn_data))
print("Number of games read in and processed: %d" % len(games))

num_games_with_errors = 0
for g in games:
    if g.error_hands > 0:
        num_games_with_errors += 1
        # print("%d hand number errors for game %s" % (g.error_hands, g.number))
print("Number of games with hand number errors: %d" % num_games_with_errors)

# TODO: concat "b" (continuation) files
