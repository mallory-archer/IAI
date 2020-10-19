import math


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
    # to protect against large values in interim steps, take difference of values, use first value as base
    exp_opt_values_f = [calc_exp_option_value(x-option_values_f[0], phi_f) for x in option_values_f]
    return [x / sum(exp_opt_values_f) for x in exp_opt_values_f]
