import os
import pandas as pd
from game_classes import *

pd.options.display.max_columns = 25

# ---- Set parameters ----
fd_data = os.path.join("..", "Data", "5H1AI_logs")  # location of data relative to code base path
fn_data = [x for x in os.listdir(fd_data) if x.find('sample_game') > -1]  # list of data file names (log files)

# ---- Define functions -----
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

# ----- map player names which appear to be mis-labeled across files that were combined above
for _, g in games.items():
    g.map_players()
_, _, _, _ = meta_game_stats(games)

# ---- Check action parsing by inspection ----
import random
import json
# t_game = random.choice(list(games.keys()))
# t_hand = random.choice(list(games[t_game].hands.keys()))
t_game = '72'   # '100'
t_hand = '8'    # '75'
print(games[t_game].hands[t_hand].hand_data)
print(json.dumps(games[t_game].hands[t_hand].actions, indent=4))
print(json.dumps(games[t_game].hands[t_hand].outcomes, indent=4))

# ----- parse by player -----
all_players = set()
for _, g in games.items():
    all_players = all_players.union(g.players)
del g

players = list()
for p_name in all_players:
    print('\n---- Processing player %s ------' % p_name)
    p = Player(p_name)
    p.add_games_info(games)
    players.append(p)
    del p
del p_name

# ----- save data -----
# import pickle
# with open("python_hand_data.pickle", 'wb') as f:
#     pickle.dump({'players': players, 'games': games}, f)
