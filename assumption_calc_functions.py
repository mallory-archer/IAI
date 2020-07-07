
def create_game_hand_index(player_f):
    game_hand_index_f = dict()
    for game_num in player_f.game_numbers:
        t_hand_numbers = list()
        for hand_num in list(player_f.actions[game_num].keys()):
            if len(player_f.actions[game_num][hand_num]) > 0:
                t_hand_numbers.append(hand_num)
        game_hand_index_f.update({game_num: t_hand_numbers})
    return game_hand_index_f


# calc probability of winning hand with premium hole cards
def calc_prob_winning(player_f, game_hand_index_f):
    win_list = list()
    win_premium_hole_cards = list()
    win_NOT_premium_hole_cards = list()
    for game_num, hands in game_hand_index_f.items():
        for hand_num in hands:
            try:
                win_list.append(1 if player_f.outcomes[game_num][hand_num] > 0 else 0)
                if player_f.odds[game_num][hand_num]:
                    win_premium_hole_cards.append(1 if (player_f.outcomes[game_num][hand_num] > 0) else 0)
                elif not player_f.odds[game_num][hand_num]:
                    win_NOT_premium_hole_cards.append(1 if (player_f.outcomes[game_num][hand_num] > 0) else 0)
            except KeyError:
                pass
    print('Out of %d hands surveyed, %3.3f were winning hands for player' % (
    len(win_list), sum(win_list) / len(win_list)))
    print('Out of %d hands surveyed, %3.3f were winning hands for player | premium hole cards' % (
    len(win_premium_hole_cards), sum(win_premium_hole_cards) / len(win_premium_hole_cards)))
    print('Out of %d hands surveyed, %3.3f were winning hands for player | NOT premium hole cards' % (
    len(win_NOT_premium_hole_cards), sum(win_NOT_premium_hole_cards) / len(win_NOT_premium_hole_cards)))

def guess_blind_amount(player_f, game_hand_index_f):
    # guess blind amount
    outcome_big = list()
    outcome_small = list()
    for game_num, hands in game_hand_index_f.items():
        for hand_num in hands:
            # --- infer blinds
            if player_f.blinds[game_num][hand_num]['big'] and (abs(player_f.outcomes[game_num][hand_num]) < 500) and (
                    player_f.outcomes[game_num][hand_num] < 0):
                outcome_big.append(player_f.outcomes[game_num][hand_num])
            if player_f.blinds[game_num][hand_num]['small'] and (abs(player_f.outcomes[game_num][hand_num]) < 500) and (
                    player_f.outcomes[game_num][hand_num] < 0):
                outcome_small.append(player_f.outcomes[game_num][hand_num])

    # calc average payoff if go beyond first round of betting
    win_play_amount = list()
    loss_play_amount = list()
    for game_num, hands in game_hand_index_f.items():
        for hand_num in hands:
            try:
                if (player_f.outcomes[game_num][hand_num] > 0) and (
                        player_f.actions[game_num][hand_num]['preflop'] != 'f'):
                    win_play_amount.append(player_f.outcomes[game_num][hand_num])
                if (player_f.outcomes[game_num][hand_num] < 0) and (
                        player_f.actions[game_num][hand_num]['preflop'] != 'f'):
                    loss_play_amount.append(player_f.outcomes[game_num][hand_num])
            except KeyError:
                pass

