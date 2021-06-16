# define poker specific functions


def calc_looseness(num_calls_f, num_raises_f, num_tot_obs_f):
    # Smith 2009: The generally accepted measure of looseness is the
    # percentage of hands in which a player voluntarily
    # puts money into the pot. This can include a call or a
    # raise, but does not include blind bets because these
    # are involuntary.
    return (num_calls_f + num_raises_f) / num_tot_obs_f


def calc_aggressiveness(num_raises_f, num_checks_f, num_calls_f):
    # Smith 2009: The generally accepted
    # measure of aggression is the ratio of the number of
    # bets and raises to the number of checks and calls.
    return num_raises_f / (num_checks_f + num_calls_f)
