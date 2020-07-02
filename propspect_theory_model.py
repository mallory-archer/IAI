# ------ Fit prospect theory model -----
import random
import numpy as np
from statsmodels.base.model import GenericLikelihoodModel
from prospect_theory_funcs import *


class Option:
    def __init__(self, name, outcomes):
        self.name = name
        self.outcomes = outcomes

    def calc_option_value(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f):
        return sum([calc_outcome_value(
            calc_outcome_v(x_f=t_outcome['payoff'], alpha_f=alpha_f, lambda_f=lambda_f, beta_f=beta_f),
            calc_outcome_pi(p_f=t_outcome['prob'], c_f=gamma_f if (t_outcome['payoff'] >= 0) else delta_f))
                    for _, t_outcome in self.outcomes.items()])


class ChoiceSituation:
    def __init__(self, sit_options, sit_choice=None):
        self.options = sit_options
        self.option_names = [x.name for x in sit_options]
        self.choice = sit_choice
        self.values = None
        self.prob_options = None
        self.prob_choice = None
        if (self.choice is not None) and self.choice not in self.option_names:
            print("Warning: specified choice of '%s' is not an option in choice set: %s" % (
            self.choice, self.option_names))

    def get_option_values(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f):
        return [x.calc_option_value(alpha_f, lambda_f, beta_f, gamma_f, delta_f) for x in self.options]

    def get_option_probs(self, phi_f):
        if self.values is not None:
            return calc_prob_options(self.values, phi_f)
        else:
            print("Error: no option values stored for choice situation. Set via method 'get_option_values(args)' first")
            return None

    def get_choice_location(self, choice_f):
        return self.option_names.index(choice_f)

    def get_prob_choice(self, choice_f):
        if self.prob_options is not None:
            return self.prob_options[self.get_choice_location(choice_f)]
        else:
            print(
                "Error: no option probabilities stored for choice situation. Set via method 'get_option_probs(args)' first")
            return None

    def set_model_values(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f, phi_f):
        self.values = self.get_option_values(alpha_f, lambda_f, beta_f, gamma_f, delta_f)
        self.prob_options = self.get_option_probs(phi_f)
        self.prob_choice = self.get_prob_choice(self.choice)


class ProspectModel(GenericLikelihoodModel):
    def __init__(self, endog, **kwds):
        super(ProspectModel, self).__init__(endog, **kwds)
        self.alpha_f = None
        self.lambda_f = None
        self.beta_f = None
        self.gamma_f = None
        self.delta_f = None

    # refresh choice sitaution values under updated model parameters
    def refresh_choice_situation_values(self):
        for t_choice_sit in self.choice_situations_f:
            t_choice_sit.set_model_values(alpha_f=self.alpha_f, lambda_f=self.lambda_f, beta_f=self.beta_f,
                                          gamma_f=self.gamma_f, delta_f=self.delta_f,
                                          phi_f=self.phi_f)

    # ---- Manual LL and score ----
    # replaces default loglikelihood for each observation
    def loglikeobs(self, params):
        # print(self.choice_probs)
        self.alpha_f = params[0]
        self.lambda_f = params[1]
        self.beta_f = params[2]
        self.gamma_f = params[3]
        self.delta_f = params[4]

        # -- for given evaluation of likelihood at current value of params, calculate choice information
        self.refresh_choice_situation_values()

        # --- calc log-likelihood
        LL_f = [np.log(x.prob_choice) for x in self.choice_situations_f]

        return np.array(LL_f)

    # replaces default gradient loglikelihood for each observation (Jacobian)
    def score_obs(self, params, **kwds):
        def calc_grad(options_ff, j_chosen_ff, prob_choice_ff, alpha_ff, lambda_ff, beta_ff, gamma_ff, delta_ff,
                      phi_ff):
            # d(LL)/d(x) = sum[(1/Pr) * d(Pr)/d(x)]
            # d(Pr)/d(x=alpha, beta, lambda) = d(Pr)/d(V) * d(V)/d(v) * d(v)/d(x=alpha, beta, lambda)
            # d(Pr)/d(x=c=gamma, delta)      = d(Pr)/d(V) * d(V)/d(pi) * d(pi)/d(x=c=gamma, delta)
            def d_v_d_alpha(x_fff, alpha_fff):
                return (x_fff ** alpha_fff) * np.log(x_fff) if (x_fff > 0) else 0

            def d_v_d_beta(x_fff, beta_fff, lambda_fff):
                return (-lambda_fff) * ((-x_fff) ** beta_fff) * np.log(-x_fff) if (x_fff < 0) else 0

            def d_v_d_lambda(x_fff, beta_fff):
                return -1 * ((-x_fff) ** beta_fff) if (x_fff < 0) else 0

            def d_pi_d_c(p_fff, c_fff):
                pc = p_fff ** c_fff
                pc1 = (1 - p_fff) ** c_fff
                t_mult = pc * ((pc + pc1) ** (-1 / c_fff))
                t_n1 = np.log(pc + pc1)
                t_d1 = c_fff ** 2
                t_n2 = (pc * np.log(p_fff)) + (pc1 * np.log(1 - p_fff))
                t_d2 = c_fff * (pc + pc1)
                t_quant = (t_n1 / t_d1) - (t_n2 / t_d2)
                return t_mult * (t_quant + np.log(p_fff))

            def d_Vcomponent_d_x(prob_ffff, payoff_ffff, select_param_ffff, alpha_ffff, lambda_ffff, beta_ffff,
                                 gamma_ffff, delta_ffff):
                # for each parameter, returns d_v_d_parameter, d_pi_d_parameter
                if select_param_ffff == 'alpha':
                    return d_v_d_alpha(x_fff=payoff_ffff, alpha_fff=alpha_ffff), 0
                if select_param_ffff == 'lambda':
                    return d_v_d_lambda(x_fff=payoff_ffff, beta_fff=beta_ffff), 0
                if select_param_ffff == 'beta':
                    return d_v_d_beta(x_fff=payoff_ffff, beta_fff=beta_ffff, lambda_fff=lambda_ffff), 0
                if select_param_ffff == 'gamma':
                    return 0, d_pi_d_c(p_fff=prob_ffff, c_fff=gamma_ffff) if (
                                payoff_ffff >= 0) else 0  # f payoff is < 0, then delta is used and gamma doesn't factor in
                if select_param_ffff == 'delta':
                    return 0, d_pi_d_c(p_fff=prob_ffff, c_fff=delta_ffff) if (
                                payoff_ffff < 0) else 0  # f payoff is >= 0, then gamma is used and delta doesn't factor in

            def d_Voutcome_d_x(v_ffff, pi_ffff, d_v_d_x_ffff, d_pi_d_x_ffff):
                # d(V) / d(param) = d(v * pi) = [d(v)/d(param) * pi] + [v * d(pi)/d(param)]
                return (d_v_d_x_ffff * pi_ffff) + (v_ffff * d_pi_d_x_ffff)

            def calc_d_Pr_d_x(options_fff, j_chosen_fff, d_param_fff, alpha_fff, lambda_fff, beta_fff, gamma_fff,
                              delta_fff, phi_fff):
                def d_V_dx(outcomes_ffff, d_param_ffff, alpha_ffff, lambda_ffff, beta_ffff, gamma_ffff, delta_ffff):
                    t_d_Voutcome_d_x = list()
                    for _, t_outcome in outcomes_ffff.items():
                        t_pi = calc_outcome_pi(t_outcome['prob'], gamma_ffff if t_outcome['payoff'] > 0 else delta_ffff)
                        t_v = calc_outcome_v(t_outcome['payoff'], alpha_ffff, lambda_ffff, beta_ffff)
                        t_d_v_d_x, t_d_pi_d_x = d_Vcomponent_d_x(prob_ffff=t_outcome['prob'],
                                                                 payoff_ffff=t_outcome['payoff'],
                                                                 select_param_ffff=d_param_ffff,
                                                                 alpha_ffff=alpha_ffff, lambda_ffff=lambda_ffff,
                                                                 beta_ffff=beta_ffff,
                                                                 gamma_ffff=gamma_ffff, delta_ffff=delta_ffff)
                        t_d_Voutcome_d_x.append(d_Voutcome_d_x(t_v, t_pi, t_d_v_d_x, t_d_pi_d_x))
                        del t_pi, t_v, t_d_v_d_x, t_d_pi_d_x
                    return sum(t_d_Voutcome_d_x)

                # calc components specific to parameter
                t_d_V_dx_options = [
                    d_V_dx(j_opt.outcomes, d_param_fff, alpha_fff, lambda_fff, beta_fff, gamma_fff, delta_fff)
                    for j_opt in options_fff]

                # calc components common across derivatives
                t_exp_option_vals = [calc_exp_option_value(
                    j_opt.calc_option_value(alpha_fff, lambda_fff, beta_fff, gamma_fff, delta_fff), phi_fff)
                                     for j_opt in options_fff]

                f_f = t_exp_option_vals[j_chosen_fff]
                g_f = sum(t_exp_option_vals)
                f_prime_f = phi_fff * t_exp_option_vals[j_chosen_fff] * t_d_V_dx_options[j_chosen_fff]
                g_prime_f = phi_fff * np.dot(t_exp_option_vals, t_d_V_dx_options)
                return (f_prime_f * g_f - g_prime_f * f_f) / (g_f ** 2)

            grad_params_f = ['alpha', 'lambda', 'beta', 'gamma', 'delta']
            grad_values_f = list()
            for t_select_param in grad_params_f:
                grad_values_f.append(
                    (1 / prob_choice_ff) *
                    calc_d_Pr_d_x(options_fff=options_ff, j_chosen_fff=j_chosen_ff, d_param_fff=t_select_param,
                                  alpha_fff=alpha_ff, lambda_fff=lambda_ff, beta_fff=beta_ff,
                                  gamma_fff=gamma_ff, delta_fff=delta_ff, phi_fff=phi_ff)
                )

            return dict(zip(grad_params_f, grad_values_f))

        # ensure that the values stores for options, probs, choice_probs are update for current value of likelihood
        self.loglikeobs(params)  # may be able to remove this to halve the amount of execution time

        jacob_f = list()
        for cs_f in model.choice_situations_f:
            t_grad = calc_grad(options_ff=cs_f.options, j_chosen_ff=cs_f.get_choice_location(cs_f.choice),
                               prob_choice_ff=cs_f.prob_choice,
                               alpha_ff=self.alpha_f, lambda_ff=self.lambda_f, beta_ff=self.beta_f,
                               gamma_ff=self.gamma_f, delta_ff=self.delta_f,
                               phi_ff=self.phi_f)  ##### self. = model., self.c --> self.gamma, self.delta
            jacob_f.append([t_grad['alpha'], t_grad['lambda'], t_grad['beta'], t_grad['gamma'], t_grad['delta']])
        return np.array(jacob_f)  # np.sum(np.array(jacob_f), axis=0)


def generate_synthetic_data(n_hands, params_actual, phi):
    choice_situations = list()
    for n in range(0, n_hands):
        t1_win_prob = random.random()
        t2_win_prob = random.random()

        def draw_payoff():
            return random.random() * 200 - 100

        t_choice_options = [Option(name='play', outcomes={'win': {'payoff': draw_payoff(), 'prob': t1_win_prob},
                                                          'lose': {'payoff': draw_payoff(), 'prob': 1 - t1_win_prob}}),
                            Option(name='fold', outcomes={'win': {'payoff': draw_payoff(), 'prob': t2_win_prob},
                                                          'lose': {'payoff': draw_payoff(), 'prob': 1 - t2_win_prob}})]
        t_choice_situation = ChoiceSituation(sit_options=t_choice_options[:], sit_choice=None)
        t_choice_situation.values = t_choice_situation.get_option_values(alpha_f=params_actual['alpha'],
                                                                         lambda_f=params_actual['lambda'],
                                                                         beta_f=params_actual['beta'],
                                                                         gamma_f=params_actual['gamma'],
                                                                         delta_f=params_actual['delta'])
        t_choice_situation.prob_options = t_choice_situation.get_option_probs(phi_f=phi)
        choice_situations.append(t_choice_situation)
        del t_choice_situation
    del n

    for t_sit in choice_situations:
        t_sit.choice = (
            success_event_name if random.random() < t_sit.get_prob_choice(success_event_name) else fail_event_name)
    del t_sit
    # -------------------------------------------------------------------------------

    # reset values created for synthetic generation back to none
    for cs in choice_situations:
        cs.values = None
        cs.prob_options = None
        cs.prob_choice = None
    return choice_situations


# ------------- SET PARAMETERS ------------------------------
phi = .5

# ------------- GENERATE SYNTHETIC DATA ----------------------
param_names_actual = ['alpha', 'lambda', 'beta', 'gamma', 'delta']
param_values_actual = [0.88, 2.25, 0.88, 0.61, 0.69]
params_actual = dict(zip(param_names_actual, param_values_actual))
n_hands = 1000
success_event_name = 'fold'
fail_event_name = 'play'
# t_choice_options = [Option(name='play', outcomes={'win': {'payoff': 100, 'prob': 0.6}, 'lose': {'payoff': -100, 'prob': 0.4}}), Option(name='fold', outcomes={'win': {'payoff': 0.01, 'prob': 0.5}, 'lose': {'payoff': -10, 'prob': 0.5}})]
choice_situations = generate_synthetic_data(n_hands, params_actual, phi)

# ------------ FIT MODEL -------------------
# Optimization parameters
maxiter = 100
pgtol = 1e-8
ftol = 1e-12
start_params = [0.5, 1, 0.5, 0.5, 0.5]
bounds_params = [(1e-2, 1 - 1e-2), (1e-2, 5), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2), (1e-2, 1 - 1e-2)]

# Plotting parameters
select_param_name1 = 'alpha'
select_param_name2 = 'lambda'
calc_contour_info = False
n_1D_mesh_points = 5

# --- Fit model
model = ProspectModel(endog=np.array([1 if x.choice == success_event_name else 0 for x in choice_situations]), choice_situations_f=choice_situations, phi_f=phi)
model_result = model.fit(start_params=start_params, method='LBFGS', maxiter=maxiter, disp=True, bounds=bounds_params, pgtol=pgtol, factr=ftol)
print(model_result.summary())
print('Actual parameters: %s' % params_actual)
print('Gradient at solution: %s' % dict(zip(param_names_actual, model.score(model_result.params))))

# =============== Graphical examination ====================
import matplotlib.pyplot as plt


def get_result_info(select_param_name_f, param_names_f, param_values_f, param_domain_f, calc_contour_info_f=False, select_param_name2_f=None, n_1D_mesh_points=10):
    p_loc_in_params = param_names_f.index(select_param_name_f)
    p_vec = np.linspace(param_domain_f[select_param_name_f][0], param_domain_f[select_param_name_f][1], 25)
    
    LL_p = list()
    score_p = list()
    for p in p_vec[1:]:
        t_params = param_values_f[0:p_loc_in_params] + [p] + param_values_f[(p_loc_in_params + 1):]
        LL_p.append(np.sum(model.loglikeobs(t_params)))
        score_p.append(np.sum(model.score_obs(np.array(t_params))[:, p_loc_in_params]))
    finite_diff_deriv_LL_p = [(LL_p[i_f] - LL_p[i_f - 1]) / (p_vec[i_f] - p_vec[i_f - 1]) for i_f in range(1, len(LL_p))]
    
    p_vec1 = None
    p_vec2 = None
    LL_matrix = None
    if calc_contour_info_f and (select_param_name2_f is not None):
        # ----- contour plot
        p_vec1 = np.linspace(param_domain_f[select_param_name_f][0], param_domain_f[select_param_name_f][1], n_1D_mesh_points)
        p_vec2 = np.linspace(param_domain_f[select_param_name2_f][0], param_domain_f[select_param_name2_f][1], n_1D_mesh_points)
        LL_matrix = np.empty(shape=[len(p_vec1), len(p_vec2)])
        p_loc1 = param_names_f.index(select_param_name_f)
        p_loc2 = param_names_f.index(select_param_name2_f)

        for i in range(0, len(p_vec1)):
            for j in range(0, len(p_vec2)):
                t_base_params = param_values_f.copy()
                t_base_params[p_loc1] = p_vec1[i]
                t_base_params[p_loc2] = p_vec2[j]
                LL_matrix[i, j] = np.sum(model.loglikeobs(t_base_params))

    return p_vec, LL_p, score_p, finite_diff_deriv_LL_p, {select_param_name_f: p_vec1}, {select_param_name2_f: p_vec2}, LL_matrix


def plot_functions(select_param_name_f, p_vec_f, LL_p_f, score_p_f=None, finite_diff_deriv_LL_p_f=None,
                   plot_LL_num_deriv=False, plot_num_analytical_deriv=False,
                   plot_contour=False, XX_f=None, YY_f=None, LL_matrix_f=None, n_levels_f=10):
    # --- Likelihood function and derivatives
    if plot_LL_num_deriv:
        # ---- LL function and numerical derivative
        fig3, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel(select_param_name_f)
        ax1.set_ylabel('LL', color=color)
        ax1.plot(p_vec_f[1:], LL_p_f, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('numerical derivative approximation', color=color)  # we already handled the x-label with ax1
        if finite_diff_deriv_LL_vec is not None:
            ax2.plot(p_vec_f[1:], finite_diff_deriv_LL_p_f[0:] + [None], color=color)
        if score_p_f is not None:
            ax2.plot(p_vec_f[1:], score_p_f, color='green')
        ax2.tick_params(axis='y', labelcolor=color)

        fig3.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    # --- Derivatives only
    if plot_num_analytical_deriv:
        fig1, ax1 = plt.subplots()
        ax1.plot(p_vec_f[1:], score_p_f, color='green')
        ax1.plot(p_vec_f[1:], [None] + finite_diff_deriv_LL_p_f, color='blue')
        ax1.set_xlabel(select_param_name_f)
        ax1.set_ylabel('Value')
        ax1.legend(['analytical', 'numerical'])
        plt.show()

        # ---- for different axes scales -----
        # fig2, ax1 = plt.subplots()
        #
        # color = 'tab:green'
        # ax1.set_xlabel(select_param_name_f)
        # ax1.set_ylabel('analytical derivative', color=color)
        # ax1.plot(p_vec_f[1:], score_p_f, color=color)
        # ax1.tick_params(axis='y', labelcolor=color)
        #
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #
        # color = 'tab:blue'
        # ax2.set_ylabel('numerical derivative approximation', color=color)  # we already handled the x-label with ax1
        # ax2.plot(p_vec_f[1:], finite_diff_deriv_LL_p_f[0:] + [None], color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        #
        # fig2.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()

    # --- Contour plot
    if plot_contour and (list(XX_f.values())[0] is not None) and (list(YY_f.values())[0] is not None):
        fig, ax1 = plt.subplots()
        CS = plt.contour(list(XX_f.values())[0], list(YY_f.values())[0], LL_matrix_f, n_levels_f)
        ax1.set_xlabel(list(XX_f.keys())[0])
        ax1.set_ylabel(list(YY_f.keys())[0])
        ax1.set_title('LL contour plot')
        cbar1 = fig.colorbar(CS)
        cbar1.ax.set_ylabel('LL')
        cbar1.add_lines(CS)
        ax1.plot(list(XX_f.values())[0][np.where(LL_matrix_f == np.nanmax(LL_matrix_f))[0]],
                 list(YY_f.values())[0][np.where(LL_matrix_f == np.nanmax(LL_matrix_f))[1]],
                 marker="o")


p_vec, LL_vec, score_vec, finite_diff_deriv_LL_vec, XX, YY, LL_matrix = get_result_info(select_param_name_f=select_param_name1, param_names_f=param_names_actual, param_values_f=param_values_actual,
                                                                              param_domain_f=dict(zip(param_names_actual, bounds_params)),
                                                                              calc_contour_info_f=calc_contour_info, select_param_name2_f=select_param_name2, n_1D_mesh_points=n_1D_mesh_points)
plot_functions('alpha', p_vec, LL_vec, score_vec, finite_diff_deriv_LL_vec, plot_LL_num_deriv=True, plot_num_analytical_deriv=True, plot_contour=True, XX_f=XX, YY_f=YY, LL_matrix_f=LL_matrix)
