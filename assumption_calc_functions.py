# summary stats on data set


def create_game_hand_index(player_f):
    game_hand_index_f = dict()
    for game_num in player_f.game_numbers:
        t_hand_numbers = list()
        for hand_num in list(player_f.actions[game_num].keys()):
            if len(player_f.actions[game_num][hand_num]) > 0:
                t_hand_numbers.append(hand_num)
        game_hand_index_f.update({game_num: t_hand_numbers})
    return game_hand_index_f


def calc_prob_winning(player_f, game_hand_index_f):
    # calc probability of winning hand with premium hole cards
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


def calc_exp_loss_wins(games_f, small_blind_f=50, big_blind_f=100):
    losses_dict = {'small_excl': {'sum': 0, 'count': 0}, 'big_excl': {'sum': 0, 'count': 0},
                   '1': {'sum': 0, 'count': 0}, '2': {'sum': 0, 'count': 0},
                   '3': {'sum': 0, 'count': 0}, '4': {'sum': 0, 'count': 0}, '5': {'sum': 0, 'count': 0},
                   '6': {'sum': 0, 'count': 0}}
    wins_dict = {'blinds_excl': {'sum': 0, 'count': 0}, '1': {'sum': 0, 'count': 0}, '2': {'sum': 0, 'count': 0},
                 '3': {'sum': 0, 'count': 0}, '4': {'sum': 0, 'count': 0}, '5': {'sum': 0, 'count': 0},
                 '6': {'sum': 0, 'count': 0}}

    for g_num in games_f.keys():
        for h_num in games_f[g_num].hands.keys():
            if (games_f[g_num].hands[h_num].small_blind != list(games_f[g_num].hands[h_num].outcomes.keys())[0]) or (
                    games_f[g_num].hands[h_num].big_blind != list(games_f[g_num].hands[h_num].outcomes.keys())[1]):
                print('ERROR: order of dictionary may not match order of seats. Check assumption_calc_functions.py')
            t_dict_keys = games_f[g_num].hands[h_num].outcomes.keys()
            for seat_num in range(0, len(games_f[g_num].hands[h_num].outcomes)):
                if games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] < 0:
                    # record losses if player chose to play the hand (excluding pre-flop fold or 0 money on the table)
                    losses_dict[str(seat_num + 1)]['sum'] = losses_dict[str(seat_num + 1)]['sum'] + \
                                                            games_f[g_num].hands[h_num].outcomes[
                                                                list(t_dict_keys)[seat_num]]
                    losses_dict[str(seat_num + 1)]['count'] = losses_dict[str(seat_num + 1)]['count'] + 1
                    if (list(t_dict_keys)[seat_num] == games_f[g_num].hands[h_num].small_blind) and (
                            games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] != (small_blind_f * -1)):
                        # if the player is the small blind and loses only the small blind amount ignore since it wasn't a decision it was compulsory
                        losses_dict['small_excl']['sum'] = losses_dict['small_excl']['sum'] + \
                                                           games_f[g_num].hands[h_num].outcomes[
                                                               list(t_dict_keys)[seat_num]]
                        losses_dict['small_excl']['count'] = losses_dict['small_excl']['count'] + 1
                    if (list(t_dict_keys)[seat_num] == games_f[g_num].hands[h_num].big_blind) and (
                            games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] != (big_blind_f * -1)):
                        # if the player is the small blind and loses only the big blind amount ignore since it wasn't a decision it was compulsory
                        losses_dict['big_excl']['sum'] = losses_dict['big_excl']['sum'] + \
                                                         games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]]
                        losses_dict['big_excl']['count'] = losses_dict['big_excl']['count'] + 1
                if games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] > 0:
                    # record wins if player chose to play the hand
                    wins_dict[str(seat_num + 1)]['sum'] = wins_dict[str(seat_num + 1)]['sum'] + \
                                                          games_f[g_num].hands[h_num].outcomes[
                                                              list(t_dict_keys)[seat_num]]
                    wins_dict[str(seat_num + 1)]['count'] = wins_dict[str(seat_num + 1)]['count'] + 1
                    if (list(t_dict_keys)[seat_num] == games_f[g_num].hands[h_num].big_blind) and (
                            games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] != (
                            small_blind_f + big_blind_f)):
                        # if the player is the big blind and collects only the small and big blind, ignore since there was no decision made by any player, all money was compulsory
                        wins_dict['blinds_excl']['sum'] = wins_dict['blinds_excl']['sum'] + \
                                                          games_f[g_num].hands[h_num].outcomes[
                                                              list(t_dict_keys)[seat_num]]
                        wins_dict['blinds_excl']['count'] = wins_dict['blinds_excl']['count'] + 1
            del t_dict_keys

    for t_pos, t_dict in wins_dict.items():
        print('Average size win for position %s: %f' % (t_pos, t_dict['sum'] / t_dict['count']))
    for t_pos, t_dict in losses_dict.items():
        print('Average size loss for position %s: %f' % (t_pos, t_dict['sum'] / t_dict['count']))
        
    return losses_dict, wins_dict


def calc_prob_winning_slansky_rank(games_f, small_blind_f=50, big_blind_f=100):
    slanksy_prob_dict_f = {'1': {'win': 0, 'count': 0}, '2': {'win': 0, 'count': 0}, '3': {'win': 0, 'count': 0},
                           '4': {'win': 0, 'count': 0}, '5': {'win': 0, 'count': 0}, '6': {'win': 0, 'count': 0},
                           '7': {'win': 0, 'count': 0}, '8': {'win': 0, 'count': 0}, '9': {'win': 0, 'count': 0}}

    for g_num in games_f.keys():
        for h_num in games_f[g_num].hands.keys():
            for p in games_f[g_num].hands[h_num].outcomes.keys():
                slanksy_prob_dict_f[str(games_f[g_num].hands[h_num].odds[p]['slansky'])]['count'] = \
                slanksy_prob_dict_f[str(games_f[g_num].hands[h_num].odds[p]['slansky'])]['count'] + 1
                if (games_f[g_num].hands[h_num].outcomes[p] > 0) and (
                        games_f[g_num].hands[h_num].outcomes[p] != (small_blind_f + big_blind_f)):
                    slanksy_prob_dict_f[str(games_f[g_num].hands[h_num].odds[p]['slansky'])]['win'] = \
                    slanksy_prob_dict_f[str(games_f[g_num].hands[h_num].odds[p]['slansky'])]['win'] + 1

    for t_slansky_rank, t_dict in slanksy_prob_dict_f.items():
        print('Prob of winning for Slansky rank %s: %3.1f%% (%d obs)' % (t_slansky_rank, t_dict['win'] / t_dict['count'] * 100, t_dict['count']))
    del t_slansky_rank, t_dict

    return slanksy_prob_dict_f
