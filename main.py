print('Hello world.')

# A = ['L', 'R']  # action state L == 'left'/curl according to right hand rule, R == 'right'/curl according to right hand rule
S = ['A', 'B', 'C', 'D']

# ---- Specify game parameters ----
p = {'A': {'L': 0, 'R': 1},
     'B': {'L': 0, 'R': 1},
     'C': {'L': 0, 'R': 1},
     'D': {'L': 0, 'R': 1}}# policy = given a state, what should the agent do? structured as {s1: {a1: frequency of choice 1, ... aN: frequency of choice N}, ... sN}

# Specify prob(s'|s, a) where dictionary is structured as ={s1':{s1:{a1: p1, a2:p1, ... aN:pN},...sN},...s'N}
P = {'A': {'A': {'L': 0, 'R': 0}, 'B': {'L': .9, 'R': .1}, 'C': {'L': 0, 'R': 1}, 'D': {'L': 0, 'R': 0}},
     'B': {'A': {'L': 0.1, 'R': 0.9}, 'B': {'L': 0, 'R': 0}, 'C': {'L': 0, 'R': 0}, 'D': {'L': 0, 'R': 0}},
     'C': {'A': {'L': .9, 'R': .1}, 'B': {'L': 0, 'R': 0}, 'C': {'L': 0, 'R': 0}, 'D': {'L': 0, 'R': 0}},
     'D': {'A': {'L': 0, 'R': 0}, 'B': {'L': 0.1, 'R': 0.9}, 'C': {'L': 0.1, 'R': 0.9}, 'D': {'L': 0, 'R': 0}}}

# Specify R(s,a) where dictionary is structured as ={s1': {s1:{a1:r1, a2:r2, ... aN:rN}},...sN'}
R = {'A': {'A': {'L': 0, 'R': 0}, 'B': {'L': -10, 'R': 0},  'C': {'L': 0, 'R': -10},  'D': {'L': 0, 'R': 0}},
     'B': {'A': {'L': 0, 'R': -10},  'B': {'L': 0, 'R': 0}, 'C': {'L': 0, 'R': 0}, 'D': {'L': 0, 'R': 0}},
     'C': {'A': {'L': -10, 'R': 0},  'B': {'L': 0, 'R': 0}, 'C': {'L': 0, 'R': 0}, 'D': {'L': 0, 'R': 0}},
     'D': {'A': {'L': 0, 'R': 0}, 'B': {'L': 0, 'R': -10},  'C': {'L': -10, 'R': 0},  'D': {'L': 100,  'R': 100}}}

disc_rate = 1

# -- Specify convergence criterion
convergence_tol = 0.1


# ----- Define functions -----
def eval_a(Vs_prime, r_sa, Psprime_sa, disc_rate):
    return Psprime_sa * (r_sa + disc_rate * Vs_prime)


def eval_Vs(s_prime, s, Vs, P, disc_rate):
    Vs_new = 0
    for a in p[s].keys():
        Vsa = eval_a(Vs[s_prime], R[s_prime][s][a], P[s_prime][s][a], disc_rate)    # contribution to state value from taking action "a" under the given policy
        Vs_new = Vs_new + p[s][a] * Vsa                                             # add part contribution of value of action under policy weighted by likelihood of taking the action in the current state under the policy
    return Vs_new


# ----- Evaluate policy -----
Vs = {'A': 50, 'B': 50, 'C': 50, 'D': 100}
Vs_temp = dict.fromkeys(Vs.keys())
delta = dict.fromkeys(Vs.keys(), 0)
max_delta = 0
# while max_delta > convergence_tol:
for i in range(1, 10):
    for s_prime in S:
        for s in p.keys():
            Vs_temp[s] = eval_Vs(s_prime, s, Vs, P, disc_rate)
            delta[s] = abs(Vs_temp[s] - Vs[s])
        Vs.update(Vs_temp)
        max_delta = max(delta.values())
        print((max_delta, Vs))



