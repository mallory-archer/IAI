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
def eval_a(Psa, R, Vs, disc_rate):
    Vsa_new = 0
    for s_prime in Psa.keys():
        if s_prime == 'terminal':
            temp = R['terminal']
        else:
            temp = (Psa[s_prime] * (R[s_prime][s] + disc_rate * Vs[s_prime]))
        Vsa_new = Vsa_new + temp
        del temp
    return Vsa_new


def eval_Vs(p, s, Vs, P, R, disc_rate):
    Vs_new = 0
    for a in p[s].keys():
        Vsa = eval_a(P[s][a], R, Vs, disc_rate)    # contribution to state value from taking action "a" under the given policy
        Vs_new = Vs_new + p[s][a] * Vsa                                             # add part contribution of value of action under policy weighted by likelihood of taking the action in the current state under the policy
    return Vs_new


# ----- Evaluate policy -----
Vs = {'A': 0, 'B': 0, 'C': 0, 'D': 0}   # Vs = {'A': 75.61, 'B': 87.56, 'C': 68.05, 'D': 100}
max_delta = 100000
while max_delta > convergence_tol:
# for i in range(0, 10):
    Vs_temp = dict.fromkeys(Vs.keys())
    delta = dict.fromkeys(Vs.keys(), 0)
    for s in S:
        Vs_temp[s] = eval_Vs(p, s, Vs, P, R, disc_rate)
        delta[s] = abs(Vs_temp[s] - Vs[s])

    Vs.update(Vs_temp)
    max_delta = max(delta.values())
    print((max_delta, Vs))
print((max_delta, Vs))



