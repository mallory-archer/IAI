import os

control_param_dict = {
    'save_TF': True,
    'fp_output': 'output',
    'fp_choice_situations': os.path.join('choice_situations'),
    'fn_payoff_dict': 'payoff_dict_dnn.json',

    # ---- Params -----
    # --- data selection params
    'select_prob_source': 'dnn_prob',   # [perfect_prob, perfect_prob_noise, 'dnn_prob', 'perfect_prob_8020', 'perfect_prob5149']
    'select_player_list': ['Joe'], # 'MrBlonde', 'Gogo', 'Bill', 'MrPink', 'ORen', 'MrBlue', 'Budd', 'MrBrown', 'MrWhite', 'Hattori'], # 'Pluribus', 'Eddie', 'MrOrange'
    'select_case': 'post_neutral_or_blind_only',    # 'post_win_excl_blind_only' 'post_neutral_or_blind_only'  #'post_loss' #'post_loss_excl_blind_only'  # options: post_loss, post_win, post_loss_excl_blind_only, post_win_excl_blind_only, post_neutral, post_neutral_or_blind_only
    'select_gamble_types': ['prob_risk_decreases_omega_increases'],
    'fraction_of_data_to_use_for_estimation': 0.8,
    
    # ---- multi start params
    'num_multistarts': 1300,
    'save_iter': 100,
    'save_index_start': 0,
    # --- model fitting parameters
    'select_model_param_names': ['omega', 'lambda'],
    'select_prob_model_type': 'RUM',
    'lb_model': {'kappa': 0.000, 'lambda': 0.0000, 'omega': 0},
    'ub_model': {'kappa': 0.25, 'lambda': 200, 'omega': 4},  # omega_max_95percentile
    'ub_additional': {'kappa': 0.25, 'lambda': 100, 'omega': 2.2},
    'ftol': 1e-12,
    'gtol': 1e-5,
    'maxiter': 500
}


class ControlParams:
    def __init__(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.create_dict_params_to_set()

    def create_dict_params_to_set(self):
        t_select_player_list_save_string = '_'.join(self.select_player_list).lower()
        setattr(self, 'fn_prob_dict', {'dnn_prob': 'prob_dict_dnn.json', 'perfect_prob': 'pred_dict_perfect.json',
                                       'perfect_prob_noise': 'pred_dict_perfect_noise.json',
                                       'perfect_prob_8020': 'pred_dict_8020.json',
                                       'perfect_prob_5149': 'pred_dict_5149.json'}[self.select_prob_source])
        setattr(self, 'select_player_list_save_string', t_select_player_list_save_string)
        setattr(self, 'choice_situations_dir_save_string', os.path.join(self.fp_output,
                                                              self.fp_choice_situations,
                                                              self.select_prob_source))
        setattr(self, 'multi_start_dir_save_string', os.path.join('output', 'iter_multistart_saves',
                                                        t_select_player_list_save_string,
                                                        self.select_prob_source,
                                                        self.select_case))
        setattr(self, 'param_estimates_dir_save_string', os.path.join('output', 'iter_multistart_saves',
                                                            t_select_player_list_save_string,
                                                            self.select_prob_source, 'est_params'))
        del t_select_player_list_save_string

    def print_params(self):
        print('------ Control param values------')
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))


results_param_dict = {'fp_output': 'output',
                      'fp_choice_situations': os.path.join('choice_situations'),
                      'select_states': ['post_loss_excl_blind_only', 'post_neutral_or_blind_only', 'post_win_excl_blind_only'],
                      'select_gamble_types': ['prob_risk_decreases_omega_increases', 'safe_dominant',
                                              'risky_dominant'],
                      'select_player_list': ['Bill', 'Eddie', 'Joe', 'MrOrange', 'MrPink', 'Pluribus', ['Eddie', 'MrOrange', 'Joe', 'MrBlonde', 'Gogo', 'Bill', 'MrPink', 'Oren', 'MrBlue', 'Budd', 'MrBrown', 'MrWhite', 'Hattori']],    #
                      'select_prob_source': 'dnn_prob',
                      'save_TF': True
                      }


class ResultsParams:
    def __init__(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.create_dict_params_to_set()

    def set_player_list_save_string(self, player_list):
        setattr(self, 'select_player_list_save_string', '_'.join(player_list).lower())

    def set_param_estimates_dir_save_string(self, player_list_save_string):
        setattr(self, 'param_estimates_dir_save_string', os.path.join('output', 'iter_multistart_saves',
                                                                      player_list_save_string,
                                                                      self.select_prob_source, 'est_params')),

    def create_dict_params_to_set(self):
        # self.set_player_list_save_string(self.select_player_list)   # written this way so can dynamically set the player list save string for results processing loop
        # self.set_param_estimates_dir_save_string(self.select_player_list_save_string)
        setattr(self, 'choice_situations_dir_save_string', os.path.join(self.fp_output,
                                                                        self.fp_choice_situations,
                                                                        self.select_prob_source)),

        setattr(self, 'results_save_string', os.path.join('output', 'results'))


    def print_params(self):
        print('------ Control param values------')
        for k, v in self.__dict__.items():
            print('%s: %s' % (k, v))


