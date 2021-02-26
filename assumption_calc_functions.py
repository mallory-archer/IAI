# summary stats on data set
import pandas as pd
from scipy.stats import norm
import numpy as np
import copy


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
                    slansky_prob_dict_f[str(rank)][str(seat)].update({str(stack): {'win': 0, 'count': 0, 'play_count': 0}})
                    slansky_payoff_dict_f[str(rank)][str(seat)].update(
                        {str(stack): {'win_sum': 0, 'loss_sum': 0, 'neutral_sum': 0, 'win_count': 0, 'loss_count': 0, 'neutral_count': 0}})
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
                        slansky_prob_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['play_count'] += 1
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['win_sum'] += games_ff[g_num].hands[h_num].outcomes[p]
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['win_count'] += 1
                    elif games_ff[g_num].hands[h_num].outcomes[p] < 0:
                        slansky_prob_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['play_count'] += 1
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['loss_sum'] += games_ff[g_num].hands[h_num].outcomes[p]
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['loss_count'] += 1
                    elif games_ff[g_num].hands[h_num].outcomes[p] == 0:
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['neutral_sum'] += games_ff[g_num].hands[h_num].outcomes[p]
                        slansky_payoff_dict_ff[str(games_ff[g_num].hands[h_num].odds[p]['slansky'])][t_seat_num][t_stack_rank]['neutral_count'] += 1

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
        # elif len(slansky_groups_ff) == 1:
        #     slansky_groups_ff = [slansky_groups_ff]

        if seat_groups_ff is None:
            seat_groups_ff = [[str(z)] for z in range(1, max([max([int(x) for x in d1.keys()]) for d1 in slansky_prob_dict_ff.values()]) + 1)]
        # elif len(seat_groups_ff) == 1:
        #     seat_groups_ff = [seat_groups_ff]

        if stack_groups_ff is None:
            stack_groups_ff = [[str(z)] for z in range(1, max(max([max([max([int(x) for x in d2.keys()]) for d2 in d1.values()] for d1 in slansky_prob_dict_ff.values())])) + 1)]
        # elif len(stack_groups_ff) == 1:
        #     stack_groups_ff = [stack_groups_ff]

        # create output dictionaries
        slansky_prob_dict_out_f = dict(zip([''.join(x) for x in slansky_groups_ff],
                                           [dict(zip([''.join(y) for y in seat_groups_ff],
                                                     [dict(zip([''.join(z) for z in stack_groups_ff],
                                                               [dict()] * len(stack_groups_ff)))
                                                      ] * len(seat_groups_ff)))
                                            ] * len(slansky_groups_ff)))
        slansky_payoff_dict_out_f = copy.deepcopy(slansky_prob_dict_out_f)
        # slansky_payoff_dict_out_f = dict(zip([''.join(x) for x in slansky_groups_ff], [dict(zip([''.join(y) for y in seat_groups_ff], [dict(zip([''.join(z) for z in stack_groups_ff], [dict() for i in stack_groups_ff])) for j in seat_groups_ff])) for k in slansky_groups_ff]))

        for rank in slansky_groups_ff:
            slansky_prob_dict_out_f[''.join(rank)] = {'win': 0, 'count': 0, 'play_count': 0}
            slansky_payoff_dict_out_f[''.join(rank)] = {'win_sum': 0, 'loss_sum': 0, 'win_count': 0, 'loss_count': 0, 'neutral_sum': 0, 'neutral_count': 0}
            for seat in seat_groups_ff:
                slansky_prob_dict_out_f[''.join(rank)][''.join(seat)] = {'win': 0, 'count': 0, 'play_count': 0}
                slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)] = {'win_sum': 0, 'loss_sum': 0, 'neutral_sum': 0, 'win_count': 0, 'loss_count': 0, 'neutral_count': 0}
                for stack in stack_groups_ff:
                    t_stack_count = sum([slansky_prob_dict_ff[x][y][z]['count'] for z in stack for y in seat for x in rank])
                    t_stack_win = sum([slansky_prob_dict_ff[x][y][z]['win'] for z in stack for y in seat for x in rank])
                    t_stack_play_count = sum([slansky_prob_dict_ff[x][y][z]['play_count'] for z in stack for y in seat for x in rank])

                    t_stack_win_sum = sum([slansky_payoff_dict_ff[x][y][z]['win_sum'] for z in stack for y in seat for x in rank])
                    t_stack_loss_sum = sum([slansky_payoff_dict_ff[x][y][z]['loss_sum'] for z in stack for y in seat for x in rank])
                    t_stack_neutral_sum = sum([slansky_payoff_dict_ff[x][y][z]['neutral_sum'] for z in stack for y in seat for x in rank])

                    t_stack_win_count = sum([slansky_payoff_dict_ff[x][y][z]['win_count'] for z in stack for y in seat for x in rank])
                    t_stack_loss_count = sum([slansky_payoff_dict_ff[x][y][z]['loss_count'] for z in stack for y in seat for x in rank])
                    t_stack_neutral_count = sum([slansky_payoff_dict_ff[x][y][z]['neutral_count'] for z in stack for y in seat for x in rank])

                    # aggregate at slansky-seat-stack level
                    slansky_prob_dict_out_f[''.join(rank)][''.join(seat)][''.join(stack)] = {'win': t_stack_win, 'count': t_stack_count, 'play_count': t_stack_play_count}
                    slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)][''.join(stack)] = {'win_sum': t_stack_win_sum, 'loss_sum': t_stack_loss_sum, 'neutral_sum': t_stack_neutral_sum,
                                                                                               'win_count': t_stack_win_count, 'loss_count': t_stack_loss_count, 'neutral_count': t_stack_neutral_count}

                    # aggregate at slansky-seat level
                    slansky_prob_dict_out_f[''.join(rank)][''.join(seat)].update({'win': slansky_prob_dict_out_f[''.join(rank)][''.join(seat)]['win'] + t_stack_win,
                                                                                  'count': slansky_prob_dict_out_f[''.join(rank)][''.join(seat)]['count'] + t_stack_count,
                                                                                  'play_count': slansky_prob_dict_out_f[''.join(rank)][''.join(seat)]['play_count'] + t_stack_play_count})
                    slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)].update({'win_sum': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['win_sum'] + t_stack_win_sum,
                                                                                    'loss_sum': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['loss_sum'] + t_stack_loss_sum,
                                                                                    'neutral_sum': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['neutral_sum'] + t_stack_neutral_sum,
                                                                                    'win_count': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['win_count'] + t_stack_win_count,
                                                                                    'loss_count': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['loss_count'] + t_stack_loss_count,
                                                                                    'neutral_count': slansky_payoff_dict_out_f[''.join(rank)][''.join(seat)]['neutral_count'] + t_stack_neutral_count})

                    # aggregate at slansky level
                    slansky_prob_dict_out_f[''.join(rank)].update({'win': slansky_prob_dict_out_f[''.join(rank)]['win'] + t_stack_win,
                                                                   'count': slansky_prob_dict_out_f[''.join(rank)]['count'] + t_stack_count,
                                                                   'play_count': slansky_prob_dict_out_f[''.join(rank)]['play_count'] + t_stack_play_count})
                    slansky_payoff_dict_out_f[''.join(rank)].update({'win_sum': slansky_payoff_dict_out_f[''.join(rank)]['win_sum'] + t_stack_win_sum,
                                                                     'loss_sum': slansky_payoff_dict_out_f[''.join(rank)]['loss_sum'] + t_stack_loss_sum,
                                                                     'neutral_sum': slansky_payoff_dict_out_f[''.join(rank)]['neutral_sum'] + t_stack_neutral_sum,
                                                                     'win_count': slansky_payoff_dict_out_f[''.join(rank)]['win_count'] + t_stack_win_count,
                                                                     'loss_count': slansky_payoff_dict_out_f[''.join(rank)]['loss_count'] + t_stack_loss_count,
                                                                     'neutral_count': slansky_payoff_dict_out_f[''.join(rank)]['neutral_count'] + t_stack_neutral_count,
                                                                     })

        # CHECK AGGREGATION
        # add original data together
        t_win_orig_check = 0
        t_count_orig_check = 0
        for rank in [x for x in slansky_prob_dict_ff.keys() if (x != 'win') & (x != 'count') & (x != 'play_count')]:
            for seat in [x for x in slansky_prob_dict_ff[rank].keys() if (x != 'win') & (x != 'count') & (x != 'play_count')]:
                for stack in [x for x in slansky_prob_dict_ff[rank][seat].keys() if (x != 'win') & (x != 'count') & (x != 'play_count')]:
                    t_win_orig_check += slansky_prob_dict_ff[rank][seat][stack]['win']
                    t_count_orig_check += slansky_prob_dict_ff[rank][seat][stack]['count']
        print('Original (unaggregated) data had %d wins over %d hands' % (t_win_orig_check, t_count_orig_check))

        # add aggregations together
        t_win_check = 0
        t_count_check = 0
        for rank in [x for x in slansky_prob_dict_out_f.keys() if (x != 'win') & (x != 'count') & (x != 'play_count')]:
            for seat in [x for x in slansky_prob_dict_out_f[rank].keys() if (x != 'win') & (x != 'count') & (x != 'play_count')]:
                for stack in [x for x in slansky_prob_dict_out_f[rank][seat].keys() if (x != 'win') & (x != 'count') & (x != 'play_count')]:
                    t_win_check += slansky_prob_dict_out_f[rank][seat][stack]['win']
                    t_count_check += slansky_prob_dict_out_f[rank][seat][stack]['count']
        print('Aggregated data has %d wins over %d hands' % (t_win_check, t_count_check))

        if (t_win_check != t_win_orig_check) or (t_count_check != t_count_orig_check):
            print('WARNING: AGGREGATED DATA DOES NOT MATCH ORIGINAL DATA, SOME DATA HAS BEEN EXCLUDED IN ERROR.')

        return slansky_prob_dict_out_f, slansky_payoff_dict_out_f

    max_slansky_rank_f, max_seat_f, max_stack_rank_f = get_max_indices(games_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = preload_dict(max_slansky_rank_f, max_seat_f, max_stack_rank_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = create_counts(games_f, slansky_prob_dict_f, slansky_payoff_dict_f)
    slansky_prob_dict_f, slansky_payoff_dict_f = marginal_aggregation(slansky_prob_dict_f, slansky_payoff_dict_f,
                                                                      slansky_groups_ff=slansky_groups_f,
                                                                      seat_groups_ff=seat_groups_f,
                                                                      stack_groups_ff=stack_groups_f
                                                                      )
    # ---- check functions for manual investigations
    # t_win = 0
    # t_count = 0
    # t_play_count = 0
    # for rank in [x for x in slansky_prob_dict_f.keys() if (x != 'win') and (x != 'count') and (x != 'play_count')]:
    #     for seat in [x for x in slansky_prob_dict_f[rank].keys() if (x != 'win') and (x != 'count') and (x != 'play_count')]:
    #         for stack in [x for x in slansky_prob_dict_f[rank][seat] if (x != 'win') and (x != 'count') and (x != 'play_count')]:
    #             t_win += slansky_prob_dict_f[rank][seat][stack]['win']
    #             t_count += slansky_prob_dict_f[rank][seat][stack]['count']
    #             t_play_count += slansky_prob_dict_f[rank][seat][stack]['play_count']

    # ---- optional dataframe creation for investigation / manipulation
    # df_f = pd.DataFrame(columns=['rank', 'seat', 'stack', 'win', 'count', 'play_count', 'win_sum', 'loss_sum', 'neutral_sum', 'win_count', 'loss_count', 'neutral_count'])
    # for rank in [x for x in slansky_prob_dict_f.keys() if (x != 'win') and (x != 'count') and (x != 'play_count')]:
    #     for seat in [x for x in slansky_prob_dict_f[rank].keys() if (x != 'win') and (x != 'count') and (x != 'play_count')]:
    #         for stack in [x for x in slansky_prob_dict_f[rank][seat] if (x != 'win') and (x != 'count') and (x != 'play_count')]:
    #             df_f = df_f.append(pd.Series({'rank': rank, 'seat': seat, 'stack': stack,
    #                                           'win': slansky_prob_dict_f[rank][seat][stack]['win'],
    #                                           'count': slansky_prob_dict_f[rank][seat][stack]['count'],
    #                                           'play_count': slansky_prob_dict_f[rank][seat][stack]['play_count'],
    #                                           'win_sum': slansky_payoff_dict_f[rank][seat][stack]['win_sum'],
    #                                           'win_count': slansky_payoff_dict_f[rank][seat][stack]['win_count'],
    #                                           'loss_sum': slansky_payoff_dict_f[rank][seat][stack]['loss_sum'],
    #                                           'loss_count': slansky_payoff_dict_f[rank][seat][stack]['loss_count'],
    #                                           'neutral_sum': slansky_payoff_dict_f[rank][seat][stack]['neutral_sum'],
    #                                           'neutral_count': slansky_payoff_dict_f[rank][seat][stack]['neutral_count']}),
    #                                ignore_index=True)
    # if (df_f.win_sum.sum() + df_f.loss_sum.sum()) != 0:
    #     print('WARNING: Total amount of wins and losses across all hands are not zero. May be an error if system is closed.')
    # if sum(df_f.win - df_f.win_count) != 0:
    #     print('WARNING: number of wins in probability dictionary does not match number of wins in payoff dictionary')
    # if sum(df_f['count'] - (df_f.win_count + df_f.loss_count + df_f.neutral_count)) != 0:
    #     print('WARNING: number of wins in probability dictionary does not match number of wins in payoff dictionary')
    #
    # t_group_cols = ['rank', 'seat']
    # t_output_cols = ['win', 'count', 'play_count', 'win_sum', 'win_count', 'loss_sum', 'loss_count']
    # df_f_grouped = df_f[t_group_cols + t_output_cols].groupby(t_group_cols).agg(dict(zip(t_output_cols, ['sum'] * len(t_output_cols))))
    # df_f_grouped['prob_win_cond_play'] = df_f_grouped['win'] / df_f_grouped['play_count']
    # df_f_grouped['avg_loss_cond_play'] = df_f_grouped['loss_sum'] / df_f_grouped['loss_count']
    # df_f_grouped['avg_win_cond_play'] = df_f_grouped['win_sum'] / df_f_grouped['win_count']
    # df_f_grouped['exp_value_play'] = df_f_grouped['prob_win_cond_play'] * df_f_grouped['avg_win_cond_play'] + \
    #                                  (1 - df_f_grouped['prob_win_cond_play']) * df_f_grouped['avg_loss_cond_play']

    return slansky_prob_dict_f, slansky_payoff_dict_f


def two_sample_test_prop(p1_f, p2_f, n1_f, n2_f, n_sides_f):
    phat_f = ((p1_f * n1_f) + (p2_f * n2_f))/(n1_f + n2_f)
    z_f = (p1_f - p2_f)/np.sqrt(phat_f * (1 - phat_f) * ((1/n1_f) + (1/n2_f)))
    p_f = (1 - norm.cdf(abs(z_f))) * n_sides_f
    return z_f, p_f


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
