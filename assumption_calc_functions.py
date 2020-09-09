# summary stats on data set
import pandas as pd


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
    t_pass_count = 0
    for game_num, hands in game_hand_index_f.items():
        for hand_num in hands:
            try:
                if player_f.outcomes[game_num][hand_num] > 0:
                    win_list.append(1)
                else:
                    win_list.append(0)

                if player_f.odds[game_num][hand_num]['both_hole_premium_cards']:
                    if player_f.outcomes[game_num][hand_num] > 0:
                        win_premium_hole_cards.append(1)
                    else:
                        win_premium_hole_cards.append(0)
            except KeyError:
                t_pass_count += 1

    print('\nOut of %d hands surveyed, %3.3f were winning hands for player %s' % (len(win_list), sum(win_list) / len(win_list), player_f.name))
    print('Out of %d hands surveyed, %3.3f were winning hands for player %s | premium hole cards' % (len(win_premium_hole_cards), sum(win_premium_hole_cards) / len(win_premium_hole_cards), player_f.name))
    print('Out of %d hands surveyed, %3.3f were winning hands for player %s | NOT premium hole cards\n' % (len(win_list) - len(win_premium_hole_cards), (len(win_premium_hole_cards) - sum(win_premium_hole_cards)) / (len(win_list) - len(win_premium_hole_cards)), player_f.name))


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


def calc_prob_winning_slansky_rank(games_f):    #, players_f, small_blind_f=50, big_blind_f=100):
    def get_max_indices(games_ff):
        # format: {'slansky rank': {'seat': {'stack rank': {'win': '#', 'count': '#'}}}}
        # get domains of slansky ranks, player seats, and stack ranks
        t_max_slansky_rank = 1
        t_max_seat = 1
        t_max_stack_rank = 1
        for g_num in games_ff.keys():
            for h_num in games_ff[g_num].hands.keys():
                for p in games_ff[g_num].hands[h_num].outcomes.keys():
                    t_max_stack_rank = max(t_max_stack_rank, games_ff[g_num].hands[h_num].start_stack_rank[p])
                    t_max_slansky_rank = max(t_max_slansky_rank, games_ff[g_num].hands[h_num].odds[p]['slansky'])
                    t_max_seat = max(t_max_seat, len(games_ff[g_num].hands[h_num].players))
        return t_max_slansky_rank, t_max_seat, t_max_stack_rank

    def preload_dict(t_max_slansky_rank_f, t_max_seat_f, t_max_stack_rank_f):
        # create dictionary placeholder
        slansky_prob_dict_f = dict()
        slansky_payoff_dict_f = dict()
        for rank in range(1, t_max_slansky_rank_f + 1):
            slansky_prob_dict_f.update({str(rank): {}})
            slansky_payoff_dict_f.update({str(rank): {}})
            for seat in range(1, t_max_seat_f + 1):
                slansky_prob_dict_f[str(rank)].update({str(seat): {}})
                slansky_payoff_dict_f[str(rank)].update({str(seat): {}})
                for stack in range(1, t_max_stack_rank_f + 1):
                    slansky_prob_dict_f[str(rank)][str(seat)].update({str(stack): {'win': 0, 'count': 0}})
                    slansky_payoff_dict_f[str(rank)][str(seat)].update(
                        {str(stack): {'win_sum': 0, 'loss_sum': 0, 'win_count': 0, 'loss_count': 0}})
                    # print('rank %d, seat %d, stack %d' % (rank, seat, stack))
        return slansky_prob_dict_f, slansky_payoff_dict_f
    
    def create_counts(games_ff, slansky_prob_dict_ff, slansky_payoff_dict_ff):
        # create counts
        for g_num in games_ff.keys():
            for h_num in games_ff[g_num].hands.keys():
                for p in games_ff[g_num].hands[h_num].outcomes.keys():
                    t_seat_num = str(games_ff[g_num].hands[h_num].players.index(p) + 1)
                    t_stack_rank = str(games_ff[g_num].hands[h_num].start_stack_rank[p])

                    # counts
                    slansky_prob_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['count'] += 1

                    # wins / losses
                    if games_ff[g_num].hands[h_num].outcomes[p] > 0:
                        slansky_prob_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['win'] += 1
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['win_sum'] += games_ff[g_num].hands[h_num].outcomes[p]
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['win_count'] += 1

                    elif games_ff[g_num].hands[h_num].outcomes[p] < 0:
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['loss_sum'] += games_ff[g_num].hands[h_num].outcomes[p]
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['loss_count'] += 1
                    # note: ignores case where player has 0 outcome, meaning opted to fold
        del g_num, h_num, p
        return slansky_prob_dict_ff, slansky_payoff_dict_ff

    def marginal_aggregation(slansky_prob_dict_ff, slansky_payoff_dict_ff):
        # aggregate over stack_size
        o_count = 0
        o_win = 0
        o_win_sum = 0
        o_loss_sum = 0
        o_win_count = 0
        o_loss_count = 0
        for rank in slansky_prob_dict_ff.keys():
            t_count = 0
            t_win = 0
            t_win_sum = 0
            t_loss_sum = 0
            t_win_count = 0
            t_loss_count = 0
            for seat in slansky_prob_dict_ff[rank].keys():
                t_seat_count = 0
                t_seat_win = 0
                t_seat_win_sum = 0
                t_seat_loss_sum = 0
                t_seat_win_count = 0
                t_seat_loss_count = 0
                for stack in slansky_prob_dict_ff[rank][seat].keys():
                    t_seat_count += slansky_prob_dict_ff[rank][seat][stack]['count']
                    t_seat_win += slansky_prob_dict_ff[rank][seat][stack]['win']
                    t_seat_win_sum += slansky_payoff_dict_ff[rank][seat][stack]['win_sum']
                    t_seat_loss_sum += slansky_payoff_dict_ff[rank][seat][stack]['loss_sum']
                    t_seat_win_count += slansky_payoff_dict_ff[rank][seat][stack]['win_count']
                    t_seat_loss_count += slansky_payoff_dict_ff[rank][seat][stack]['loss_count']
                slansky_prob_dict_ff[rank][seat].update({'win': t_seat_win, 'count': t_seat_count})
                slansky_payoff_dict_ff[rank][seat].update({'win_sum': t_seat_win_sum, 'loss_sum': t_seat_loss_sum, 'win_count': t_seat_win_count, 'loss_count': t_seat_loss_count})

                # must be completed after aggregation of stack such that 'count', 'win', 'win_sum', 'loss_sum', and 'obs_count' exist at seat level
                t_count += slansky_prob_dict_ff[rank][seat]['count']
                t_win += slansky_prob_dict_ff[rank][seat]['win']
                t_win_sum += slansky_payoff_dict_ff[rank][seat]['win_sum']
                t_loss_sum += slansky_payoff_dict_ff[rank][seat]['loss_sum']
                t_win_count += slansky_payoff_dict_ff[rank][seat]['win_count']
                t_loss_count += slansky_payoff_dict_ff[rank][seat]['loss_count']
            slansky_prob_dict_ff[rank].update({'win': t_win, 'count': t_count})
            slansky_payoff_dict_ff[rank].update({'win_sum': t_win_sum, 'loss_sum': t_loss_sum, 'win_count': t_win_count, 'loss_count': t_loss_count})

            # must be completed after aggregation of seat such that 'count', 'win', 'win_sum', 'loss_sum', and 'obs_count' exist at rank level
            o_count += slansky_prob_dict_ff[rank]['count']
            o_win += slansky_prob_dict_ff[rank]['win']
            o_win_sum += slansky_payoff_dict_ff[rank]['win_sum']
            o_loss_sum += slansky_payoff_dict_ff[rank]['loss_sum']
            o_win_count += slansky_payoff_dict_ff[rank]['win_count']
            o_loss_count += slansky_payoff_dict_ff[rank]['loss_count']
        slansky_prob_dict_ff.update({'win': o_win, 'count': o_count})
        slansky_payoff_dict_ff.update({'win_sum': o_win_sum, 'loss_sum': o_loss_sum, 'win_count': o_win_count, 'loss_count': o_loss_count})

        return slansky_prob_dict_ff, slansky_payoff_dict_ff

    def create_aggregate_prob_payoff_dicts(slansky_prob_dict_ff, slansky_payoff_dict_ff):
        prob_rank_dict_f = dict()
        payoff_rank_dict_f = dict()
        prob_rank_seat_dict_f = dict()
        payoff_rank_seat_dict_f = dict()
        prob_rank_seat_stack_dict_f = dict()
        payoff_rank_seat_stack_dict_f = dict()
        for rank in slansky_prob_dict_ff.keys():
            try:
                int(rank)  # ignore summary keys like "win" and "count"
                rank_info_prob = slansky_prob_dict_ff[rank]
                rank_info_payoff = slansky_payoff_dict_ff[rank]
                prob_rank_dict_f.update({rank: {}})
                payoff_rank_dict_f.update({rank: {}})
                prob_rank_seat_dict_f.update({rank: {}})
                payoff_rank_seat_dict_f.update({rank: {}})
                prob_rank_seat_stack_dict_f.update({rank: {}})
                payoff_rank_seat_stack_dict_f.update({rank: {}})
                for seat in slansky_prob_dict_ff[rank].keys():
                    try:
                        int(seat)
                        rank_seat_info_prob = rank_info_prob[seat]
                        rank_seat_info_payoff = rank_info_payoff[seat]
                        prob_rank_seat_dict_f[rank].update({seat: {}})
                        payoff_rank_seat_dict_f[rank].update({seat: {}})
                        prob_rank_seat_stack_dict_f[rank].update({seat: {}})
                        payoff_rank_seat_stack_dict_f[rank].update({seat: {}})
                        for stack in slansky_prob_dict_ff[rank][seat].keys():
                            try:
                                int(stack)
                                rank_seat_stack_info_prob = rank_seat_info_prob[stack]
                                rank_seat_stack_info_payoff = rank_seat_info_payoff[stack]
                                prob_rank_seat_stack_dict_f[rank][seat].update(
                                    {stack: rank_seat_stack_info_prob['win'] / rank_seat_stack_info_prob['count']})
                                payoff_rank_seat_stack_dict_f[rank][seat].update({stack: {
                                    'avg_win': rank_seat_stack_info_payoff['win_sum'] / rank_seat_stack_info_payoff[
                                        'win_count'],
                                    'avg_loss': rank_seat_stack_info_payoff['loss_sum'] / rank_seat_stack_info_payoff[
                                        'loss_count']}})
                                del rank_seat_stack_info_prob, rank_seat_stack_info_payoff
                            except (ValueError, ZeroDivisionError):
                                pass
                        del stack
                        prob_rank_seat_dict_f[rank].update(
                            {seat: rank_seat_info_prob['win'] / rank_seat_info_prob['count']})
                        payoff_rank_seat_dict_f[rank].update(
                            {seat: {'avg_win': rank_seat_info_payoff['win_sum'] / rank_seat_info_payoff['win_count'],
                                    'avg_loss': rank_seat_info_payoff['loss_sum'] / rank_seat_info_payoff[
                                        'loss_count']}})
                        del rank_seat_info_prob, rank_seat_info_payoff
                    except ValueError:
                        pass
                del seat
                prob_rank_dict_f.update({rank: rank_info_prob['win'] / rank_info_prob['count']})
                payoff_rank_dict_f.update({rank: {'avg_win': rank_info_payoff['win_sum'] / rank_info_payoff['win_count'],
                                                'avg_loss': rank_info_payoff['loss_sum'] / rank_info_payoff[
                                                    'loss_count']}})
                del rank_info_prob, rank_info_payoff
            except ValueError:
                pass
        del rank
        return prob_rank_dict_f, payoff_rank_dict_f, prob_rank_seat_dict_f, payoff_rank_seat_dict_f, prob_rank_seat_stack_dict_f, payoff_rank_seat_stack_dict_f

    def format_as_dataframe(max_rank_f, max_seat_f, max_stack_rank_f, slansky_prob_dict_ff, slansky_payoff_dict_ff, print_dfs=False):
        # can probably comment out dfs and store probabilities as dict
        slansky_dfs_f = list()
        slansky_payoff_win_dfs_f = list()
        slansky_payoff_loss_dfs_f = list()
        for rank in range(1, max_rank_f + 1):
            t_df1 = pd.DataFrame(index=[str(i) for i in range(1, max_seat_f + 1)], columns=[str(j) for j in range(1, max_stack_rank_f + 1)])
            t_df2 = pd.DataFrame(index=[str(i) for i in range(1, max_seat_f + 1)], columns=[str(j) for j in range(1, max_stack_rank_f + 1)])
            t_df3 = pd.DataFrame(index=[str(i) for i in range(1, max_seat_f + 1)], columns=[str(j) for j in range(1, max_stack_rank_f + 1)])
            for seat in range(1, max_seat_f + 1):
                for stack in range(1, max_stack_rank_f + 1):
                    try:
                        t_df1.loc[str(seat), str(stack)] = slansky_prob_dict_ff[str(rank)][str(seat)][str(stack)]['win'] / slansky_prob_dict_ff[str(rank)][str(seat)][str(stack)]['count']
                    except ZeroDivisionError:
                        print('All games, no observations for str(rank) %s str(seat) %s str(stack) %s' % (str(rank), str(seat), str(stack)))
                    try:
                        t_df2.loc[str(seat), str(stack)] = slansky_payoff_dict_ff[str(rank)][str(seat)][str(stack)]['win_sum'] / slansky_payoff_dict_ff[str(rank)][str(seat)][str(stack)]['obs_count']
                    except ZeroDivisionError:
                        print('Win payoff instances, no observations for str(rank) %s str(seat) %s str(stack) %s' % (str(rank), str(seat), str(stack)))
                    try:
                        t_df3.loc[str(seat), str(stack)] = slansky_payoff_dict_ff[str(rank)][str(seat)][str(stack)]['loss_sum'] / slansky_payoff_dict_ff[str(rank)][str(seat)][str(stack)]['obs_count']
                    except ZeroDivisionError:
                        print('Loss payoff instances, no observations for str(rank) %s str(seat) %s str(stack) %s' % (str(rank), str(seat), str(stack)))
            slansky_dfs_f.append(t_df1)
            slansky_payoff_win_dfs_f.append(t_df2)
            slansky_payoff_loss_dfs_f.append(t_df3)

            if print_dfs:
                print('ALL perc. of wins for slansky rank %s by seat (rows) and stack rank (columns):' % rank)
                print(t_df1)
                print('\n')
                del t_df1

                print('Avg. winnings for slansky rank %s by seat (rows) and stack rank (columns):' % rank)
                print(t_df2)
                print('\n')
                del t_df2

                print('Avg. losses for slansky rank %s by seat (rows) and stack rank (columns):' % rank)
                print(t_df3)
                print('\n')
                del t_df3

        # create csv to examine in Tableau
        t_df = pd.DataFrame(columns=['slansky_rank', 'seat', 'stack_rank', 'count', 'wins', 'percent'])
        for slansky_rank in range(0, len(slansky_dfs_f)):
            for seat in range(1, 7):
                for stack_rank in range(1, 10):
                    t_df = t_df.append({'slansky_rank': slansky_rank + 1,
                                        'seat': seat,
                                        'stack_rank': stack_rank,
                                        'count': slansky_prob_dict_f[str(slansky_rank + 1)][str(seat)][str(stack_rank)]['count'],
                                        'wins': slansky_prob_dict_f[str(slansky_rank + 1)][str(seat)][str(stack_rank)]['win'],
                                        'percent': slansky_dfs_f[slansky_rank].loc[str(seat), str(stack_rank)],
                                        'avg. winnings': slansky_payoff_win_dfs_f[slansky_rank].loc[str(seat), str(stack_rank)],
                                        'avg. losses': slansky_payoff_loss_dfs_f[slansky_rank].loc[str(seat), str(stack_rank)]}, ignore_index=True)

        return t_df, slansky_dfs_f, slansky_payoff_win_dfs_f, slansky_payoff_loss_dfs_f

    max_slansky_rank_f, max_seat_f, max_stack_rank_f = get_max_indices(games_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = preload_dict(max_slansky_rank_f, max_seat_f, max_stack_rank_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = create_counts(games_f, slansky_prob_dict_f, slansky_payoff_dict_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = marginal_aggregation(slansky_prob_dict_f, slansky_payoff_dict_f)
    prob_rank_dict, payoff_rank_dict, prob_rank_seat_dict, payoff_rank_seat_dict, prob_rank_seat_stack_dict, payoff_rank_seat_stack_dict = create_aggregate_prob_payoff_dicts(slansky_prob_dict_f, slansky_payoff_dict_f)
    
    # Options: write to dataframe for export for Tableau examination
    # tableau_df_f, _, _, _ = format_as_dataframe(max_slansky_rank_f, max_seat_f, max_stack_rank_f, slansky_prob_dict_f, slansky_payoff_dict_f)
    # tableau_df_f.to_csv('Slansky odds examination.csv')

    return slansky_prob_dict_f, slansky_payoff_dict_f, prob_rank_dict, payoff_rank_dict, prob_rank_seat_dict, payoff_rank_seat_dict, prob_rank_seat_stack_dict, payoff_rank_seat_stack_dict

# ------ archive -------
# def calc_exp_loss_wins(games_f, small_blind_f=50, big_blind_f=100):
#     losses_dict = {'small_excl': {'sum': 0, 'count': 0}, 'big_excl': {'sum': 0, 'count': 0},
#                    '1': {'sum': 0, 'count': 0}, '2': {'sum': 0, 'count': 0},
#                    '3': {'sum': 0, 'count': 0}, '4': {'sum': 0, 'count': 0}, '5': {'sum': 0, 'count': 0},
#                    '6': {'sum': 0, 'count': 0}}
#     wins_dict = {'blinds_excl': {'sum': 0, 'count': 0}, '1': {'sum': 0, 'count': 0}, '2': {'sum': 0, 'count': 0},
#                  '3': {'sum': 0, 'count': 0}, '4': {'sum': 0, 'count': 0}, '5': {'sum': 0, 'count': 0},
#                  '6': {'sum': 0, 'count': 0}}
#
#     for g_num in games_f.keys():
#         for h_num in games_f[g_num].hands.keys():
#             if (games_f[g_num].hands[h_num].small_blind != list(games_f[g_num].hands[h_num].outcomes.keys())[0]) or (
#                     games_f[g_num].hands[h_num].big_blind != list(games_f[g_num].hands[h_num].outcomes.keys())[1]):
#                 print('ERROR: order of dictionary may not match order of seats. Check assumption_calc_functions.py')
#             t_dict_keys = games_f[g_num].hands[h_num].outcomes.keys()
#             for seat_num in range(0, len(games_f[g_num].hands[h_num].outcomes)):
#                 if games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] < 0:
#                     # record losses if player chose to play the hand (excluding pre-flop fold or 0 money on the table)
#                     losses_dict[str(seat_num + 1)]['sum'] = losses_dict[str(seat_num + 1)]['sum'] + \
#                                                             games_f[g_num].hands[h_num].outcomes[
#                                                                 list(t_dict_keys)[seat_num]]
#                     losses_dict[str(seat_num + 1)]['count'] = losses_dict[str(seat_num + 1)]['count'] + 1
#                     if (list(t_dict_keys)[seat_num] == games_f[g_num].hands[h_num].small_blind) and (
#                             games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] != (small_blind_f * -1)):
#                         # if the player is the small blind and loses only the small blind amount ignore since it wasn't a decision it was compulsory
#                         losses_dict['small_excl']['sum'] = losses_dict['small_excl']['sum'] + \
#                                                            games_f[g_num].hands[h_num].outcomes[
#                                                                list(t_dict_keys)[seat_num]]
#                         losses_dict['small_excl']['count'] = losses_dict['small_excl']['count'] + 1
#                     if (list(t_dict_keys)[seat_num] == games_f[g_num].hands[h_num].big_blind) and (
#                             games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] != (big_blind_f * -1)):
#                         # if the player is the small blind and loses only the big blind amount ignore since it wasn't a decision it was compulsory
#                         losses_dict['big_excl']['sum'] = losses_dict['big_excl']['sum'] + \
#                                                          games_f[g_num].hands[h_num].outcomes[
#                                                              list(t_dict_keys)[seat_num]]
#                         losses_dict['big_excl']['count'] = losses_dict['big_excl']['count'] + 1
#                 if games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] > 0:
#                     # record wins if player chose to play the hand
#                     wins_dict[str(seat_num + 1)]['sum'] = wins_dict[str(seat_num + 1)]['sum'] + \
#                                                           games_f[g_num].hands[h_num].outcomes[
#                                                               list(t_dict_keys)[seat_num]]
#                     wins_dict[str(seat_num + 1)]['count'] = wins_dict[str(seat_num + 1)]['count'] + 1
#                     if (list(t_dict_keys)[seat_num] == games_f[g_num].hands[h_num].big_blind) and (
#                             games_f[g_num].hands[h_num].outcomes[list(t_dict_keys)[seat_num]] != (
#                             small_blind_f + big_blind_f)):
#                         # if the player is the big blind and collects only the small and big blind, ignore since there was no decision made by any player, all money was compulsory
#                         wins_dict['blinds_excl']['sum'] = wins_dict['blinds_excl']['sum'] + \
#                                                           games_f[g_num].hands[h_num].outcomes[
#                                                               list(t_dict_keys)[seat_num]]
#                         wins_dict['blinds_excl']['count'] = wins_dict['blinds_excl']['count'] + 1
#             del t_dict_keys
#
#     for t_pos, t_dict in wins_dict.items():
#         print('Average size win for position %s: %f' % (t_pos, t_dict['sum'] / t_dict['count']))
#     for t_pos, t_dict in losses_dict.items():
#         print('Average size loss for position %s: %f' % (t_pos, t_dict['sum'] / t_dict['count']))
#
#     return losses_dict, wins_dict
