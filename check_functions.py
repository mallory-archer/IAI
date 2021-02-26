# set of functions to sanity check data / coding
import pandas as pd


# --- slansky rank classification check --- 
def slansky_sanity_check(games_f):
    num_games_f = 0
    num_hands = 0
    num_pocket_cards = 0
    rank_count = dict(zip(range(1, 10), [0] * 9))
    rank_hands = dict(zip(range(1, 10), [list()] * 9))
    for g in games_f.keys():
        num_games_f += 1
        for h in games_f[g].hands.keys():
            num_hands += 1
            for p in games_f[g].hands[h].odds.keys():
                num_pocket_cards += 1
                rank_count[games_f[g].hands[h].odds[p]['slansky']] += 1
                rank_hands[games_f[g].hands[h].odds[p]['slansky']] = rank_hands[games_f[g].hands[h].odds[p]['slansky']] + [games_f[g].hands[h].cards['hole_cards'][p]]

    rank_hands_classified_count = [len(v) for k, v in rank_hands.items()]

    print('A total of %d sets of hole cards have been classified' % sum(rank_hands_classified_count))

    if num_pocket_cards != sum(rank_hands_classified_count):
        print('WARNING: creation of rank_hands (observed rank assigned to pocket cards dictionary) has different number of classified sets than number of pairs in game')

    print('Obs. relative frequency of Slansky rank:')
    for k, v in rank_hands.items():
        print('Rank %d: %3.1f%%' % (k, len(v)/sum(rank_hands_classified_count) * 100))

    # build a dataframe of all unique hole cards observed in data set for use with subsequent manual checking lines
    # t_df = pd.DataFrame(columns=['rank', 'v1', 'v2', 's1', 's2'])
    # for r in rank_hands.keys():
    #     for i in set(rank_hands[r]):
    #         t_row =  dict(zip(['v1', 's1', 'v2', 's2'], i))
    #         t_row['rank'] = r
    #         t_df = t_df.append(t_row, ignore_index=True)
    # t_df['suited'] = (t_df['s1'] == t_df['s2'])
    # t_df.sort_values(['rank', 'suited', 's1', 's2'], inplace=True)
    # t_df.reindex()

    # this line can be used to manually / visually check which cards occur with which others in our data
    # t_df.loc[t_df['v1'].isin(['A']) & ~t_df.suited]['v2'].unique()

    # for a given game and hand, prints players cards and resulting classification
    # for p in games_f[g].hands[h].cards['hole_cards'].keys():
    #     print('%s: %s %d' % (p, games_f[g].hands[h].cards['hole_cards'][p], games_f[g].hands[h].odds[p]['slansky']))

