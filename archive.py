import numpy as np
from numpy.random import random_sample

print('Hello world.')

# A = ['L', 'R']  # action state L == 'left'/curl according to right hand rule, R == 'right'/curl according to right hand rule
S = ['A', 'B', 'C', 'D']

# ---- Specify game parameters ----
p = {'A': {'L': 0.6, 'R': 0.4},
     'B': {'L': 0, 'R': 1},
     'C': {'L': 1, 'R': 0},
     'D': {'L': 0, 'R': 1}} # policy = given a state, what should the agent do? structured as {s1: {a1: frequency of choice 1, ... aN: frequency of choice N}, ... sN}

# Probability that taking an action in a given state results in the state s_prime, structured as {s1: {a1: {s1_prime: prob of occurence ... sM_prime}...aN}...sN}
P = {'A': {'L': {'C': 0.9, 'B': 0.1}, 'R': {'B': 0.9, 'C': 0.1}},
     'B': {'L': {'A': 0.9, 'D': 0.1}, 'R': {'D': 0.9, 'A': 0.1}},
     'C': {'L': {'D': 0.9, 'A': 0.1}, 'R': {'A': 0.9, 'D': 0.1}},
     'D': {'L': {'terminal': 1}, 'R': {'terminal': 1}}}

# Specify reward for arriving at s_prime coming from s, structure as {s1_prime: {s1: r1...sN: rN}...sN_prime}
R = {'A': {'B': -10, 'C': -10},
     'B': {'A': -10},
     'C': {'A': -10},
     'D': {'B': -10, 'C': -10},
     'terminal': 100}

disc_rate = 1

# -- Specify convergence criterion
convergence_tol = 0.0001

# ----- Define functions -----
def eval_a(Psa_eaf, s_eaf, R_eaf, Vs_eaf, disc_rate_eaf):
    Vsa_new = 0
    for s_prime in Psa_eaf.keys():
        if s_prime == 'terminal':
            temp = R_eaf['terminal']
        else:
            temp = (Psa_eaf[s_prime] * (R_eaf[s_prime][s_eaf] + disc_rate_eaf * Vs_eaf[s_prime]))
        Vsa_new = Vsa_new + temp
        del temp
    return Vsa_new


def eval_Vs(p_evf, s_evf, Vs_evf, R_evf, disc_rate_evf):
    Vs_new = 0
    for a in p_evf[s_evf].keys():
        Vsa = eval_a(P[s_evf][a], s_evf, R_evf, Vs_evf, disc_rate_evf)    # contribution to state value from taking action "a" under the given policy
        Vs_new = Vs_new + p[s_evf][a] * Vsa                                             # add part contribution of value of action under policy weighted by likelihood of taking the action in the current state under the policy
    return Vs_new


def eval_Qsa(Vs_eqf, P_eqf, R_eqf, disc_rate_eqf):
    Qsa_new = dict.fromkeys(P_eqf.keys())
    for s_temp in P_eqf.keys():
        Qsa_new[s_temp] = dict.fromkeys(P_eqf[s_temp].keys())
        for a in P_eqf[s_temp].keys():
            Qsa_new[s_temp][a] = eval_a(P_eqf[s_temp][a], s_temp, R_eqf, Vs_eqf, disc_rate_eqf)
    del s_temp
    return Qsa_new

# ----- Evaluate policy -----
Vs = {'A': 0, 'B': 0, 'C': 0, 'D': 0}   # Vs = {'A': 75.61, 'B': 87.56, 'C': 68.05, 'D': 100}
max_delta = 100000
while max_delta > convergence_tol:
# for i in range(0, 10):
    Vs_temp = dict.fromkeys(Vs.keys())
    delta = dict.fromkeys(Vs.keys(), 0)
    for s in S:
        Vs_temp[s] = eval_Vs(p, s, Vs, R, disc_rate)
        delta[s] = abs(Vs_temp[s] - Vs[s])

    Vs.update(Vs_temp)
    max_delta = max(delta.values())
del max_delta, delta, Vs_temp, s

# ----- Optimize policy -----
convergence_tol_p = 0.001
p_star = {'A': {'R': .5, 'L': .5}, 'C': {'R': .25, 'L': .75}, 'D': {'R': .75, 'L': .25}, 'B': {'R': .5, 'L': .5}}   #dict(p)
max_delta = 10000
while max_delta > convergence_tol_p:
# for i in range(0, 10):
    Qsa = eval_Qsa(Vs, P, R, disc_rate)    # set current policy values
    p_star_new = dict.fromkeys(p_star.keys())
    delta = dict.fromkeys(p_star.keys(), max_delta)
    for s in Qsa.keys():
        p_star_new[s] = dict.fromkeys(Qsa[s].keys(), 0)

        p_star_new[s][max(Qsa[s], key=Qsa[s].get)] = 1

        delta_s_temp = {x: p_star_new[s][x] - p_star[s][x] for x in p_star_new[s] if x in p_star[s]}
        delta[s] = max({k: abs(v) for k, v in delta_s_temp.items()}.values())

    p_star.update(p_star_new)
    max_delta = max(delta.values())
    del p_star_new, delta, delta_s_temp

print('Optimal policy under known model (probabilities and rewards): ' + str(p_star))

# ----- Model free methods -----
# ------ MCMC Vs estimation
def run_MCMC(s0_MCMC, length, P, R, p):
    s = s0_MCMC
    sN = list(s)  # store states
    aN = list()  # store actions
    rN = list() # store rewards
    # Run an MC chain and store rewards
    for i in range(0, length):
        a = np.random.choice(a=list(p[s].keys()), size=1, replace=True, p=list(p[s].values()))[0]  # for current state s, draw action randomly from available options p[s].keys()  # p is policy but also specifies what actions are available in each state
        # ---- THIS IS HIDDEN TO AGENT (specification of P and R) -----
        s_prime = np.random.choice(a=list(P[s][a].keys()), size=1, replace=True, p=list(P[s][a].values()))[0]  # observe reward (return reward based on P distribution, which is hidden to agent but mimics what the agent would see if playing in the world
        if s_prime == 'terminal':
            r = R[s_prime]
            sN.append(s_prime)
            aN.append(a)
            rN.append(r)
            del a, s_prime, r
            return sN, aN, rN
        else:
            r = R[s_prime][s]
            sN.append(s_prime)
            aN.append(a)
            rN.append(r)
            s = s_prime
            del a, s_prime, r
        # ----------------------------------
    return sN, aN, rN


def calc_MCMC_Vs(sK, rK, Vs_0, Vs_count0):
    Vs = Vs_0
    Vs_count = Vs_count0
    for j in range(0, len(rK)):
        s = sK[j]
        Vs_count[s] = Vs_count[s] + 1
        Vs[s] = Vs[s] + (1 / Vs_count[s]) * (sum(rK[j:len(rK)]) - Vs[s])
    del s
    return Vs, Vs_count

max_k = 100     # maximum number of steps to take before terminating chain without reaching terminal node
MCMC_N = 10000     # maximum number of chains to run
Vs_MCMC = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
Vs_MCMC_count = dict.fromkeys(Vs, 0)
for i in range(0, MCMC_N):
    s0 = np.random.choice(list(p.keys()))       # randomly chosen starting state
    sK, aK, rK = run_MCMC(s0, max_k, P, R, p)      # run single chain to terminal

    Vs_MCMC, Vs_MCMC_count = calc_MCMC_Vs(sK, rK, Vs_MCMC, Vs_MCMC_count)    # evaluate single chain
del i
print('MCMC estimated Vs under given policy: ' + str(Vs_MCMC))

# ----- TD Vs estimation
def calc_Vs_TD(r, s, s_prime, disc_rate, Vs_TD, Vs_TD_count):
    return


TD_V_max_k = 10000
Vs_TD = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
Vs_TD_count = dict.fromkeys(Vs, 0)
s = 'A'
for k in range(0, TD_V_max_k):
    Vs_TD_count[s] = Vs_TD_count[s] + 1
    alpha = 1 / Vs_TD_count[s]
    a = np.random.choice(a=list(p[s].keys()), size=1, replace=True, p=list(p[s].values()))[0]  # for current state s, draw action randomly from available options p[s].keys()  # p is policy but also specifies what actions are available in each state
    # ---- THIS IS HIDDEN TO AGENT (specification of P and R) -----
    # get reward and new state
    s_prime = np.random.choice(a=list(P[s][a].keys()), size=1, replace=True, p=list(P[s][a].values()))[0]  # observe reward (return reward based on P distribution, which is hidden to agent but mimics what the agent would see if playing in the world # draw a

    if s_prime == 'terminal':
        r = R[s_prime]
        Vs_TD[s] = (1 - alpha) * Vs_TD[s] + alpha * r  # update Vs
        s = np.random.choice(a=list(p.keys()), size=1, replace=True)[0]
    else:
        r = R[s_prime][s]
        Vs_TD[s] = (1 - alpha) * Vs_TD[s] + alpha * (r + disc_rate * Vs_TD[s_prime])  # update Vs
        s = s_prime

print(Vs_TD)

# ---- Q-Learning
Q = dict.fromkeys(p.keys(), 0)
Q_count = dict.fromkeys(Q.keys(), 0)
s = 'A'
max_Q_iter = 1
for j in range(0, max_Q_iter):
    eval_Qsa(Vs_eqf, P_eqf, R_eqf, disc_rate_eqf)