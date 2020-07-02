import numpy as np
import matplotlib.pyplot as plt
import math

class Option:
    def __init__(self, name, outcomes):
        self.name = name
        self.outcomes = outcomes

    def calc_option_value(self, alpha_f, lambda_f, beta_f, gamma_f, delta_f):
        return sum([calc_outcome_value(
            calc_outcome_v(x_f=t_outcome['payoff'], alpha_f=alpha_f, lambda_f=lambda_f, beta_f=beta_f),
            calc_outcome_pi(p_f=t_outcome['prob'], c_f=gamma_f if (t_outcome['payoff'] >= 0) else delta_f))
                    for _, t_outcome in self.outcomes.items()])


def calc_outcome_v(x_f, alpha_f, lambda_f, beta_f):
    if x_f >= 0:
        return x_f ** alpha_f
    if x_f < 0:
        return -lambda_f * ((-x_f) ** beta_f)


def calc_outcome_pi(p_f, c_f):
    return (p_f ** c_f) / (((p_f ** c_f) + ((1 - p_f) ** c_f)) ** (1 / c_f))


def calc_outcome_value(v_f, pi_f):
    return v_f * pi_f


def calc_exp_option_value(option_value_f, phi_f):
    return math.exp(phi_f * option_value_f)


def calc_prob_options(option_values_f, phi_f):
    # return [x / sum(option_values_f) for x in option_values_f]

    # to protect against large values in interim steps, take difference of values, use first value as base
    exp_opt_values_f = [calc_exp_option_value(x - option_values_f[0], phi_f) for x in option_values_f]
    return [x / sum(exp_opt_values_f) for x in exp_opt_values_f]


def d_v_d_alpha(x_fff, alpha_fff):
    return (x_fff ** alpha_fff) * np.log(x_fff) if (x_fff > 0) else 0


def d_v_d_beta(x_fff, beta_fff, lambda_fff):
    return (-lambda_fff) * ((-x_fff) ** beta_fff) * np.log(-x_fff) if (x_fff < 0) else 0


def d_v_d_lambda(x_fff, beta_fff):
    return -1 * ((-x_fff) ** beta_fff) if (x_fff < 0) else 0


def d_pi_d_c(p_fff, c_fff):
    # (p_fff ** c_fff) * ((2 - (p_fff ** c_fff)) ** (-1 / c_fff)) * \
    # ((np.log(2 - (p_fff ** c_fff)) / (c_fff ** 2)) +
    #  (((p_fff ** c_fff) * np.log(p_fff)) / (c_fff * (2 - (p_fff ** c_fff))))) + \
    # (p_fff ** c_fff) * ((2 - (p_fff ** c_fff)) ** (-1 / c_fff)) * np.log(p_fff)
    pc = p_fff ** c_fff
    pc1 = (1 - p_fff) ** c_fff
    t_mult = pc * ((pc + pc1) ** (-1 / c_fff))
    t_n1 = np.log(pc + pc1)
    t_d1 = c_fff ** 2
    t_n2 = (pc * np.log(p_fff)) + (pc1 * np.log(1 - p_fff))
    t_d2 = c_fff * (pc + pc1)
    t_quant = (t_n1 / t_d1) - (t_n2 / t_d2)
    return t_mult * (t_quant + np.log(p_fff))


def d_Vcomponent_d_x(prob_ffff, payoff_ffff, select_param_ffff, alpha_ffff, lambda_ffff, beta_ffff, gamma_ffff, delta_ffff):
    # for each parameter, returns d_v_d_parameter, d_pi_d_parameter
    if select_param_ffff == 'alpha':
        return d_v_d_alpha(x_fff=payoff_ffff, alpha_fff=alpha_ffff), 0
    if select_param_ffff == 'lambda':
        return d_v_d_lambda(x_fff=payoff_ffff, beta_fff=beta_ffff), 0
    if select_param_ffff == 'beta':
        return d_v_d_beta(x_fff=payoff_ffff, beta_fff=beta_ffff, lambda_fff=lambda_ffff), 0
    if select_param_ffff == 'c':
        return 0, d_pi_d_c(p_fff=prob_ffff, c_fff=gamma_ffff if (payoff_ffff >= 0) else delta_ffff)


def d_Voutcome_d_x(v_ffff, pi_ffff, d_v_d_x_ffff, d_pi_d_x_ffff):
    # d(V) / d(param) = d(v * pi) = [d(v)/d(param) * pi] + [v * d(pi)/d(param)]
    return (d_v_d_x_ffff * pi_ffff) + (v_ffff * d_pi_d_x_ffff)


def d_V_dx(outcomes_ffff, d_param_ffff, alpha_ffff, lambda_ffff, beta_ffff, gamma_ffff, delta_ffff):
    t_d_Voutcome_d_x = list()
    for _, t_outcome in outcomes_ffff.items():
        t_pi = calc_outcome_pi(t_outcome['prob'], gamma_ffff if t_outcome['payoff'] > 0 else delta_ffff)
        t_v = calc_outcome_v(t_outcome['payoff'], alpha_ffff, lambda_ffff, beta_ffff)
        t_d_v_d_x, t_d_pi_d_x = d_Vcomponent_d_x(prob_ffff=t_outcome['prob'], payoff_ffff=t_outcome['payoff'],
                                                 select_param_ffff=d_param_ffff,
                                                 alpha_ffff=alpha_ffff, lambda_ffff=lambda_ffff, beta_ffff=beta_ffff,
                                                 gamma_ffff=gamma_ffff, delta_ffff=delta_ffff)
        t_d_Voutcome_d_x.append(d_Voutcome_d_x(t_v, t_pi, t_d_v_d_x, t_d_pi_d_x))
        del t_pi, t_v, t_d_v_d_x, t_d_pi_d_x
    return sum(t_d_Voutcome_d_x)


def calc_d_Pr_d_x(options_fff, j_chosen_fff, d_param_fff, alpha_fff, lambda_fff, beta_fff, gamma_fff, delta_fff, phi_fff):
    # calc components specific to parameter
    t_d_V_dx_options = [d_V_dx(j_option.outcomes, d_param_fff, alpha_fff, lambda_fff, beta_fff, gamma_fff, delta_fff) for j_option in options_fff]

    # calc components common across derivatives
    t_exp_option_vals = [calc_exp_option_value(j_opt.calc_option_value(alpha_fff, lambda_fff, beta_fff, gamma_fff, delta_fff), phi_fff) for j_opt in options_fff]

    f_f = t_exp_option_vals[j_chosen_fff]
    g_f = sum(t_exp_option_vals)
    f_prime_f = phi_fff * t_exp_option_vals[j_chosen_fff] * t_d_V_dx_options[j_chosen_fff]
    g_prime_f = phi_fff * np.dot(t_exp_option_vals, t_d_V_dx_options)
    return (f_prime_f * g_f - g_prime_f * f_f) / (g_f ** 2)


param_domain = {'alpha': (0, 1), 'lambda': (0, 5), 'beta': (0, 1), 'c': (0, 1)}
params_actual = {'alpha': 0.88, 'lambda': 2.25, 'beta': 0.88, 'c': 0.61}
options = [Option(name='play', outcomes={'win': {'payoff': 100, 'prob': 0.6}, 'lose': {'payoff': -100, 'prob': 0.4}}),
           Option(name='fold', outcomes={'win': {'payoff': 10, 'prob': 0.5}, 'lose': {'payoff': 0, 'prob': 0.5}})]
option = options[0]
t_j_chosen = 1
t_phi = 0.1


t_param_name = 'alpha'
# payoff = 100
# prob_payoff = 0.5
num_increments = 100
p_vec = np.linspace(param_domain[t_param_name][0], param_domain[t_param_name][1], num_increments)

# payoff = outcome['payoff']
# prob_payoff = outcome['prob']

f = list()
f_prime = list()
for p in p_vec:
    t_alpha = p if t_param_name == 'alpha' else params_actual['alpha']
    t_lambda = p if t_param_name == 'lambda' else params_actual['lambda']
    t_beta = p if t_param_name == 'beta' else params_actual['beta']
    t_gamma = p if t_param_name == 'c' else params_actual['c']
    t_delta = p if t_param_name == 'c' else params_actual['c']

    # ------------ d_v_dx, d_pi_dx ------------------------

    # --- alpha, lambda, beta
    # f.append(calc_outcome_v(x_f=payoff, alpha_f=t_alpha, lambda_f=t_lambda, beta_f=t_beta))

    # f_prime.append(d_v_d_alpha(x_fff=payoff, alpha_fff=t_alpha))
    # f_prime.append(d_v_d_beta(x_fff=payoff, beta_fff=t_beta, lambda_fff=t_lambda))
    # f_prime.append(d_v_d_lambda(x_fff=payoff, beta_fff=t_beta))

    # --- gamma, delta
    # f.append(calc_outcome_pi(p_f=prob_payoff, c_f=t_gamma))
    # f_prime.append(d_pi_d_c(p_fff=prob_payoff, c_fff=t_gamma))

    # ---------- d_outcome_d_x -------------------------
    # pi = calc_outcome_pi(prob_payoff, t_gamma if payoff > 0 else t_delta)
    # v = calc_outcome_v(payoff, t_alpha, t_lambda, t_beta)
    # print(v)
    #
    # f.append(calc_outcome_value(v, pi))
    #
    # d_v_d_x, d_pi_d_x = d_Vcomponent_d_x(prob_ffff=prob_payoff, payoff_ffff=payoff, select_param_ffff=t_param_name,
    #                                      alpha_ffff=t_alpha, lambda_ffff=t_lambda, beta_ffff=t_beta,
    #                                      gamma_ffff=t_gamma, delta_ffff=t_delta)
    # f_prime.append(d_Voutcome_d_x(v, pi, d_v_d_x, d_pi_d_x))

    # --------- d_option_d_x ------------------
    # f.append(option.calc_option_value(t_alpha, t_lambda, t_beta, t_gamma, t_delta))
    # f_prime.append(d_V_dx(option.outcomes, t_param_name, t_alpha, t_lambda, t_beta, t_gamma, t_delta))

    # --------- d_P_d_x ------------------------
    f.append(calc_prob_options([option.calc_option_value(t_alpha, t_lambda, t_beta, t_gamma, t_delta) for option in options], t_phi)[t_j_chosen])
    f_prime.append(calc_d_Pr_d_x(options, t_j_chosen, t_param_name, t_alpha, t_lambda, t_beta, t_gamma, t_delta, t_phi))


f_prime_numerical = [(f[i] - f[i-1])/(p_vec[i] - p_vec[i-1]) for i in range(1, len(f))]

# ---- function and analytical derivative ----
fig1, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel(t_param_name)
ax1.set_ylabel('value function', color=color)
ax1.plot(p_vec, f, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('deriv value function wrt ' + t_param_name, color=color)  # we already handled the x-label with ax1
ax2.plot(p_vec, f_prime, color=color)
ax2.plot(p_vec, [None] + f_prime_numerical, color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig1.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# --- Derivatives only
fig2, ax1 = plt.subplots()

ax1.plot(p_vec, f_prime, color='blue')
ax1.plot(p_vec, [None] + f_prime_numerical, color='green')
ax1.set_xlabel(t_param_name)
ax1.set_ylabel('Value')
ax1.legend(['analytical', 'numerical'])

plt.show()
