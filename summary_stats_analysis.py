from poker_funcs import calc_aggressiveness, calc_looseness
from assumption_calc_functions import two_sample_test_prop


def get_stats(cases_f, prop_stat_sig_comp_f):
    for tn, t_df in cases_f.items():
        print('Case: %s, looseness = %3.2f, aggressiveness = % 3.2f, num obs = %d' %
              (tn,
               calc_looseness(num_calls_f=t_df['num_preflop_call'].sum(),
                              num_raises_f=t_df['num_preflop_raise'].sum(),
                              num_tot_obs_f=t_df.shape[0]),
               calc_aggressiveness(num_raises_f=t_df['num_preflop_raise'].sum(),
                                   num_checks_f=t_df['num_preflop_check'].sum(),
                                   num_calls_f=t_df['num_preflop_call'].sum()),
               t_df.shape[0]
               )
              )

    for t_pair in prop_stat_sig_comp_f:
        t_tstat, t_pval = two_sample_test_prop(calc_looseness(num_calls_f=cases_f[t_pair[0]]['num_preflop_call'].sum(),
                                                              num_raises_f=cases_f[t_pair[0]][
                                                                  'num_preflop_raise'].sum(),
                                                              num_tot_obs_f=cases_f[t_pair[0]].shape[0]),
                                               calc_looseness(num_calls_f=cases_f[t_pair[1]]['num_preflop_call'].sum(),
                                                              num_raises_f=cases_f[t_pair[1]][
                                                                  'num_preflop_raise'].sum(),
                                                              num_tot_obs_f=cases_f[t_pair[1]].shape[0]),
                                               cases_f[t_pair[0]].shape[0],
                                               cases_f[t_pair[1]].shape[0],
                                               n_sides_f=2)
        print('Diff in looseness for cases %s and %s, t-stat: %3.2f p-value: %3.2f' % (t_pair[0],
                                                                                       t_pair[1],
                                                                                       t_tstat,
                                                                                       t_pval
                                                                                       )
              )
        if t_pval < 0.1:
            if t_tstat < 0:
                print('%s = more loose (voluntarily bets more)' % t_pair[1])
            else:
                print('%s = less loose (voluntarily bets less)' % t_pair[1])
        del t_tstat, t_pval
    del t_pair


def compare_poker_summary_stats(df_data_f):
    for t_player in df_data_f.player.unique():
        print('\nPlayer: %s' % t_player)
        get_stats({'all': df_data_f.loc[df_data_f.player == t_player],
                   'neutral': df_data_f.loc[(df_data_f.player == t_player) & (~df_data_f.loss_outcome_large_previous_TF) & (
                       ~df_data_f.win_outcome_large_previous_TF)],
                   'large loss': df_data_f.loc[(df_data_f.player == t_player) & (df_data_f.loss_outcome_large_previous_TF)],
                   'large win': df_data_f.loc[(df_data_f.player == t_player) & (df_data_f.win_outcome_large_previous_TF)]
                   },
                  prop_stat_sig_comp_f=[('neutral', 'large loss'), ('neutral', 'large win')]
                  )
