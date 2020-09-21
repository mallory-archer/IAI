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


def calc_prob_winning_slansky_rank(games_f, slansky_groups_f=None, seat_groups_f=None, stack_groups_f=None):    #, players_f, small_blind_f=50, big_blind_f=100):
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

    def marginal_aggregation(slansky_prob_dict_ff, slansky_payoff_dict_ff,
                             slansky_groups_ff=None,
                             seat_groups_ff=None,
                             stack_groups_ff=None):

        # If groupings are not provided, i.e. count slansky ranks 1, 2, and 3 as one aggregate group, then treat each rank as an individual
        if slansky_groups_ff is None:
            slansky_groups_ff = [[k] for k in slansky_prob_dict_ff.keys()]
        if seat_groups_ff is None:
            seat_groups_ff = [[str(z)] for z in range(1, max([max([int(x) for x in d1.keys()]) for d1 in slansky_prob_dict_ff.values()]) + 1)]
        if stack_groups_ff is None:
            stack_groups_ff = [[str(z)] for z in range(1, max(max([max([max([int(x) for x in d2.keys()]) for d2 in d1.values()] for d1 in slansky_prob_dict_ff.values())])) + 1)]

        # create output dictionaries
        slansky_prob_dict_out_f = dict(zip([''.join(x) for x in slansky_groups_ff], [dict(zip([''.join(y) for y in seat_groups_ff], [dict(zip([''.join(z) for z in stack_groups_ff], [dict() for i in stack_groups_ff])) for j in seat_groups_ff])) for k in slansky_groups_ff]))
        slansky_payoff_dict_out_f = dict(zip([''.join(x) for x in slansky_groups_ff], [dict(zip([''.join(y) for y in seat_groups_ff], [dict(zip([''.join(z) for z in stack_groups_ff], [dict() for i in stack_groups_ff])) for j in seat_groups_ff])) for k in slansky_groups_ff]))

        for rank in slansky_groups_ff:  ##### slansky_prob_dict_ff.keys():
            slansky_prob_dict_out_f[''.join(rank)] = {'win': 0, 'count': 0}
            slansky_payoff_dict_out_f[''.join(rank)] = {'win_sum': 0, 'loss_sum': 0, 'win_count': 0, 'loss_count': 0}
            for seat in seat_groups_ff:     ##### slansky_prob_dict_ff[rank].keys():
                slansky_prob_dict_out_f[''.join(rank)][''.join(seat)] = {'win': 0, 'count': 0}
                slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)] = {'win_sum': 0, 'loss_sum': 0, 'win_count': 0, 'loss_count': 0}
                for stack in stack_groups_ff:   ##### slansky_prob_dict_ff[rank][seat].keys():
                    t_stack_count = sum([slansky_prob_dict_ff[x][y][z]['count'] for z in stack for y in seat for x in rank])  # t_seat_count += slansky_prob_dict_ff[rank][seat][stack]['count']
                    t_stack_win = sum([slansky_prob_dict_ff[x][y][z]['win'] for z in stack for y in seat for x in rank])    # slansky_prob_dict_ff[rank][seat][stack]['win']
                    t_stack_win_sum = sum([slansky_payoff_dict_ff[x][y][z]['win_sum'] for z in stack for y in seat for x in rank])  # slansky_payoff_dict_ff[rank][seat][stack]['win_sum']
                    t_stack_loss_sum = sum([slansky_payoff_dict_ff[x][y][z]['loss_sum'] for z in stack for y in seat for x in rank])    # slansky_payoff_dict_ff[rank][seat][stack]['loss_sum']
                    t_stack_win_count = sum([slansky_payoff_dict_ff[x][y][z]['win_count'] for z in stack for y in seat for x in rank])    # slansky_payoff_dict_ff[rank][seat][stack]['win_count']
                    t_stack_loss_count = sum([slansky_payoff_dict_ff[x][y][z]['loss_count'] for z in stack for y in seat for x in rank])   # slansky_payoff_dict_ff[rank][seat][stack]['loss_count']

                    # aggregate at slansky-seat-stack level
                    slansky_prob_dict_out_f[''.join(rank)][''.join(seat)][''.join(stack)] = {'win': t_stack_win, 'count': t_stack_count} # slansky_prob_dict_ff[rank][seat].update({'win': t_seat_win, 'count': t_seat_count})
                    slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)][''.join(stack)] = {'win_sum': t_stack_win_sum, 'loss_sum': t_stack_loss_sum, 'win_count': t_stack_win_count, 'loss_count': t_stack_loss_count} # slansky_payoff_dict_ff[rank][seat].update({'win_sum': t_seat_win_sum, 'loss_sum': t_seat_loss_sum, 'win_count': t_seat_win_count, 'loss_count': t_seat_loss_count})

                    # aggregate at slansky-seat level
                    slansky_prob_dict_out_f[''.join(rank)][''.join(seat)].update({'win': slansky_prob_dict_out_f[''.join(rank)][''.join(seat)]['win'] + t_stack_win,
                                                                                  'count': slansky_prob_dict_out_f[''.join(rank)][''.join(seat)]['count'] + t_stack_count})
                    slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)].update({'win_sum': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['win_sum'] + t_stack_win_sum,
                                                                                    'loss_sum': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['loss_sum'] + t_stack_loss_sum,
                                                                                    'win_count': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['win_count'] + t_stack_win_count,
                                                                                    'loss_count': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['loss_count'] + t_stack_loss_count})

                    # aggregate at slansky level
                    slansky_prob_dict_out_f[''.join(rank)].update({'win': slansky_prob_dict_out_f[''.join(rank)]['win'] + t_stack_win,
                                                                   'count': slansky_prob_dict_out_f[''.join(rank)]['count'] + t_stack_count})
                    slansky_payoff_dict_out_f[''.join(rank)].update({'win_sum': slansky_payoff_dict_out_f[''.join(rank)]['win_sum'] + t_stack_win_sum,
                                                                     'loss_sum': slansky_payoff_dict_out_f[''.join(rank)]['loss_sum'] + t_stack_loss_sum,
                                                                     'win_count': slansky_payoff_dict_out_f[''.join(rank)]['win_count'] + t_stack_win_count,
                                                                     'loss_count': slansky_payoff_dict_out_f[''.join(rank)]['loss_count'] + t_stack_loss_count})

        return slansky_prob_dict_out_f, slansky_payoff_dict_out_f

    def create_aggregate_prob_payoff_dicts(slansky_prob_dict_ff, slansky_payoff_dict_ff):
        prob_dict_f = dict()
        payoff_dict_f = dict()
        
        # probabilities
        for rank in set(slansky_prob_dict_ff.keys()):
            for r in rank:
                try:
                    prob_dict_f.update({r: {'prob_win': slansky_prob_dict_ff[rank]['win'] / slansky_prob_dict_ff[rank]['count']}})
                except ZeroDivisionError:
                    pass
                for seat in set(slansky_prob_dict_ff[rank].keys()) - set(['win', 'count']):
                    for s in seat:
                        try:
                            prob_dict_f[r].update({s: {'prob_win': slansky_prob_dict_ff[rank][seat]['win'] / slansky_prob_dict_ff[rank][seat]['count']}})
                        except ZeroDivisionError:
                            pass
                        for stack in set(slansky_prob_dict_ff[rank][seat].keys()) - set(['win', 'count']):
                            for t in stack:
                                try:
                                    prob_dict_f[r][s].update({t: {'prob_win': slansky_prob_dict_ff[rank][seat][stack]['win'] / slansky_prob_dict_ff[rank][seat][stack]['count']}})
                                except ZeroDivisionError:
                                    pass

        # payoffs
        for rank in set(slansky_payoff_dict_ff.keys()):
            for r in rank:
                try:
                    payoff_dict_f.update({r: {'avg_win': slansky_payoff_dict_ff[rank]['win_sum'] / slansky_payoff_dict_ff[rank]['win_count'],
                                              'avg_loss': slansky_payoff_dict_ff[rank]['loss_sum'] / slansky_payoff_dict_ff[rank]['loss_count']}})
                except ZeroDivisionError:
                    pass
                for seat in set(slansky_payoff_dict_ff[rank].keys()) - set(['win_sum', 'win_count', 'loss_sum', 'loss_count']):
                    for s in seat:
                        try:
                            payoff_dict_f[r].update({s: {'avg_win': slansky_payoff_dict_ff[rank][seat]['win_sum'] / slansky_payoff_dict_ff[rank][seat]['win_count'],
                                                         'avg_loss': slansky_payoff_dict_ff[rank][seat]['loss_sum'] / slansky_payoff_dict_ff[rank][seat]['loss_count']}})
                        except ZeroDivisionError:
                            pass
                        for stack in set(slansky_payoff_dict_ff[rank][seat].keys()) - set(['win_sum', 'win_count', 'loss_sum', 'loss_count']):
                            for t in stack:
                                try:
                                    payoff_dict_f[r][s].update({t: {'avg_win': slansky_payoff_dict_ff[rank][seat][stack]['win_sum'] / slansky_payoff_dict_ff[rank][seat][stack]['win_count'],
                                                                    'avg_loss': slansky_payoff_dict_ff[rank][seat][stack]['loss_sum'] / slansky_payoff_dict_ff[rank][seat][stack]['loss_count']}})
                                except ZeroDivisionError:
                                    pass
        return prob_dict_f, payoff_dict_f

    def format_obs_as_dataframe(games_ff):
        df_ff = pd.DataFrame(columns=['slansky_rank', 'seat', 'stack_rank', 'preflop_fold_TF', 'payoff'])
        game_counter = 1
        tot_num_games = len(games_ff.keys())
        for g_num in games_ff.keys():
            print('Processing game %s of %s total'% (game_counter, tot_num_games))
            for h_num in games_ff[g_num].hands.keys():
                for p in games_ff[g_num].hands[h_num].players:
                    df_ff = df_ff.append({'player': p,
                                          'slansky_rank': str(games_ff[g_num].hands[h_num].odds[p]['slansky']),
                                          'seat': str(games_ff[g_num].hands[h_num].players.index(p) + 1),
                                          'stack_rank': str(games_ff[g_num].hands[h_num].start_stack_rank[p]),
                                          'preflop_fold_TF': (games_ff[g_num].hands[h_num].actions['preflop'][p] == 'f'),
                                          'payoff': games_ff[g_num].hands[h_num].outcomes[p]},
                                         ignore_index=True)
            game_counter += 1
        return df_ff

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

    def format_probs_payoffs_as_dataframe(prob_dict_ff=None, payoff_dict_ff=None):
        def count_level(d):
            return max(count_level(v) if isinstance(v, dict) else 0 for v in d.values()) + 1

        if prob_dict_ff is not None:
            try:
                # probabilities
                t_df_prob_f = pd.DataFrame.from_dict({(i, j, k): prob_dict_ff[i][j][k]
                                                      for i in prob_dict_ff.keys()
                                                      for j in prob_dict_ff[i].keys()
                                                      for k in prob_dict_ff[i][j].keys()},
                                                     orient='index').reset_index()
                t_df_prob_f[['slansky_rank', 'seat', 'stack_rank']] = pd.DataFrame(t_df_prob_f['index'].tolist(),
                                                                                   index=t_df_prob_f.index)
                t_df_prob_f.drop(columns=['index'], inplace=True)
                t_df_prob_f.rename(columns={0: 'prob_rank'}, inplace=True)
            except AttributeError:
                try:
                    t_df_prob_f = pd.DataFrame.from_dict({(i, j): prob_dict_ff[i][j]
                                                          for i in prob_dict_ff.keys()
                                                          for j in prob_dict_ff[i].keys()},
                                                         orient='index').reset_index()
                    t_df_prob_f[['slansky_rank', 'seat']] = pd.DataFrame(t_df_prob_f['index'].tolist(),
                                                                         index=t_df_prob_f.index)
                    t_df_prob_f.drop(columns=['index'], inplace=True)
                    t_df_prob_f.rename(columns={0: 'prob_rank_seat'}, inplace=True)
                except AttributeError:
                    try:
                        t_df_prob_f = pd.DataFrame.from_dict({i: prob_dict_ff[i]
                                                              for i in prob_dict_ff.keys()},
                                                             orient='index').reset_index()
                        t_df_prob_f.rename(columns={0: 'prob_rank_seat_stack', 'index': 'slansky_rank'}, inplace=True)
                    except:
                        print('Could not process probability dictionaries to data frame')
                        t_df_prob_f = None
        else:
            t_df_prob_f = None

        if payoff_dict_ff is not None:
            n_levels_ff = count_level(payoff_dict_ff)
            if n_levels_ff == 4:
                # payoffs
                t_df_payoff_f = pd.DataFrame.from_dict({(i, j, k): payoff_dict_ff[i][j][k]
                                                        for i in payoff_dict_ff.keys()
                                                        for j in payoff_dict_ff[i].keys()
                                                        for k in payoff_dict_ff[i][j].keys()},
                                                       orient='index').reset_index()
                t_df_payoff_f.rename(columns={'level_0': 'slansky_rank', 'level_1': 'seat', 'level_2': 'stack_rank',
                                              'avg_win': 'avg_win_rank_seat_stack',
                                              'avg_loss': 'avg_loss_rank_seat_stack'}, inplace=True)
            elif n_levels_ff == 3:
                # payoffs
                t_df_payoff_f = pd.DataFrame.from_dict({(i, j): payoff_dict_ff[i][j]
                                                        for i in payoff_dict_ff.keys()
                                                        for j in payoff_dict_ff[i].keys()},
                                                       orient='index').reset_index()
                t_df_payoff_f.rename(columns={'level_0': 'slansky_rank', 'level_1': 'seat',
                                              'avg_win': 'avg_win_rank_seat',
                                              'avg_loss': 'avg_loss_rank_seat'},
                                     inplace=True)
            elif n_levels_ff == 2:
                # payoffs
                t_df_payoff_f = pd.DataFrame.from_dict({i: payoff_dict_ff[i]
                                                        for i in payoff_dict_ff.keys()},
                                                       orient='index').reset_index()
                t_df_payoff_f.rename(columns={'index': 'slansky_rank',
                                              'avg_win': 'avg_win_rank',
                                              'avg_loss': 'avg_loss_rank'},
                                     inplace=True)
            else:
                t_df_payoff_f = None
        else:
            t_df_payoff_f = None
        return t_df_prob_f, t_df_payoff_f

    def calc_exp_utility(df_ff, prob_col_name_ff, win_col_name_ff, lose_col_name_ff):
        return (df_ff[prob_col_name_ff] * df_ff[win_col_name_ff]) + ((1 - df_ff[prob_col_name_ff]) * df_ff[lose_col_name_ff])

    max_slansky_rank_f, max_seat_f, max_stack_rank_f = get_max_indices(games_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = preload_dict(max_slansky_rank_f, max_seat_f, max_stack_rank_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = create_counts(games_f, slansky_prob_dict_f, slansky_payoff_dict_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = marginal_aggregation(slansky_prob_dict_f, slansky_payoff_dict_f,
                                                                      slansky_groups_ff=slansky_groups_f,
                                                                      seat_groups_ff=seat_groups_f,
                                                                      stack_groups_ff=stack_groups_f
                                                                      )
    prob_dict, payoff_dict = create_aggregate_prob_payoff_dicts(slansky_prob_dict_f, slansky_payoff_dict_f)

    # ----- Options: write to dataframe for export for Tableau examination -----
    # ----- Aggregate data frame (counts and sums)
    # tableau_df_f, _, _, _ = format_as_dataframe(max_slansky_rank_f, max_seat_f, max_stack_rank_f, slansky_prob_dict_f, slansky_payoff_dict_f)
    # tableau_df_f.to_csv('Slansky odds examination.csv')

    # ---- Disaggregated observations at game-hand-player level
    # tableau_df2_f = format_obs_as_dataframe(games_f)
    # # add calculations of probabilities and payoffs
    # for t_dicts in [(prob_rank_dict, payoff_rank_dict), (prob_rank_seat_dict, payoff_rank_seat_dict), (prob_rank_seat_stack_dict, payoff_rank_seat_stack_dict)]:
    #     t_prob_df_f, t_payoff_df_f = format_probs_payoffs_as_dataframe(*t_dicts)
    #     t_comb_dfs_f = t_prob_df_f.merge(t_payoff_df_f, how='left', on=list(set(t_prob_df_f.columns).intersection(set(t_payoff_df_f.columns))))
    #     tableau_df2_f = tableau_df2_f.merge(t_comb_dfs_f, how='left', on=list(set(tableau_df2_f).intersection(set(t_comb_dfs_f.columns))))
    #
    # # add exp utility value of playing (folding is either 0 or blind, which can be inferred by seat number)
    # for util_colname_f, t_cols in {'play_util_rank_seat_stack': ['prob_rank_seat_stack', 'avg_win_rank_seat_stack', 'avg_loss_rank_seat_stack'],
    #                                'play_util_rank_seat': ['prob_rank_seat', 'avg_win_rank_seat', 'avg_loss_rank_seat'],
    #                                'play_util_rank': ['prob_rank', 'avg_win_rank', 'avg_loss_rank']}.items():
    #     tableau_df2_f[util_colname_f] = calc_exp_utility(tableau_df2_f, *t_cols)
    # # tableau_df2_f.to_csv('observation_data_frame.csv')

    return slansky_prob_dict_f, slansky_payoff_dict_f, prob_dict, payoff_dict

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
