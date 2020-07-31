import json
from odds_functions import slansky_strength, chen_strength, exp_val_implied_prob, est_prob_slansky
from params import slansky_prob_dict
from scipy.stats import rankdata


class Hand:
    def __init__(self, raw_hand_string):
        self.hand_data = raw_hand_string
        self.number = self.get_hand_number()
        self.players = self.get_players()
        self.small_blind = self.get_small_blind()
        self.big_blind = self.get_big_blind()
        self.cards = self.get_cards()
        self.odds = self.get_odds()
        self.actions = self.get_actions()
        self.outcomes = self.get_outcomes()
        self.missing_fields = list()
        self.start_stack = None  # calculated in Game object because it is based on change log of previous hands
        self.start_stack_rank = None  # calculated in Game object because it is based on start_stack, which is change log of previous hands

        # check for initialization of select attributes
        self.check_player_completeness()

    def get_hand_number(self):
        try:
            t_hand_number = self.hand_data.split(':')[1]
            return t_hand_number
        except IndexError:
            return None

    def get_players(self):
        return [x.rstrip() for x in self.hand_data.split(':')[-1].split('|')]

    def get_small_blind(self):
        try:
            return self.players[0]
        except IndexError:
            pass

    def get_big_blind(self):
        try:
            return self.players[1]
        except IndexError:
            pass

    def get_cards(self):
        try:
            t_all_cards = self.hand_data.split(':')[3].split('|')
            if len(t_all_cards) > len(self.players):
                t_board_cards = t_all_cards[-1].split('/')
                t_hole_cards = t_all_cards[:-1] + [
                    t_board_cards.pop(0)]  # last set of hole cards splits to board because of "/" "|" convention
                return {'hole_cards': dict(zip(self.players, t_hole_cards)), 'board_cards': t_board_cards}
            else:
                t_hole_cards = t_all_cards
                return {'hole_cards': dict(zip(self.players, t_hole_cards))}
        except IndexError:
            return None

    def get_odds(self):
        try:
            t_cards = self.cards['hole_cards']
            premium_cards_f = {'A', 'K', 'Q', 'J'}
            odds_dict_f = dict()
            for k, v in t_cards.items():
                t_dict = dict()
                try:
                    # flag premium hole cards
                    t_dict.update({'both_hole_premium_cards': (v[0] in premium_cards_f) & (v[2] in premium_cards_f)})
                except:
                    pass
                try:
                    # get Chen hand strength
                    t_dict.update({'chen': chen_strength(v[0:4])})
                except:
                    pass
                try:
                    # get Sklansky hand strength
                    t_dict.update({'slansky': slansky_strength(v[0:4])})
                except:
                    pass
                # try:
                #     # get implied prob of winning from expected values observed on online poker website
                #     t_dict.update({'online_prob': exp_val_implied_prob(v[0:4])})
                # except:
                #     pass
                try:
                    # get implied prob of winning derived from this data set, slansky ranks, and hand outcomes
                    t_dict.update({'slansky_prob': est_prob_slansky(slansky_strength(v[0:4]), slansky_prob_dict)})
                except:
                    pass

                if len(t_dict) > 0:
                    odds_dict_f.update({k: t_dict})

                del t_dict

            if len(odds_dict_f) > 0:
                return odds_dict_f
            else:
                return None
        except TypeError:
            return None

    def get_actions(self):
        def get_round_action(round_actors_f, round_actions_f):
            round_dict_f = dict(zip(round_actors_f, [x for x in round_actions_f if x in {'f', 'r', 'c'}]))
            [t_actors.remove(a) for a in [k for k, v in round_dict_f.items() if v == 'f']]
            return round_dict_f

        try:
            t_actions = dict(zip(['preflop', 'flop', 'river', 'turn'], self.hand_data.split(':')[2].split('/')))
            t_actors = self.players[:]

            # adjust preflop actions to account for all folds defaulting to big blind gets pot; label as "call" for big blind
            if (len(t_actions['preflop']) < len(t_actors)) and (all([x == 'f' for x in t_actions['preflop']])):
                t_actions['preflop'] += 'c'

            action_dict_f = {'preflop': get_round_action(round_actors_f=t_actors[2:] + t_actors[0:2],
                                                         round_actions_f=t_actions[
                                                             'preflop'])}  # preflop has different order of betting
            [action_dict_f.update({k: get_round_action(round_actors_f=t_actors, round_actions_f=v)}) for k, v in
             t_actions.items() if k != 'preflop']
            return action_dict_f
        except IndexError:
            return None

    def get_outcomes(self):
        try:
            return dict(zip(self.players, [float(x) for x in self.hand_data.split(':')[4].split('|')]))
        except IndexError:
            return None

    def check_player_completeness(self, check_atts_ff=None):
        try:
            if check_atts_ff is None:
                check_atts_ff = {'cards': ['hole_cards'], 'odds': [], 'actions': ['preflop'], 'outcomes': []}
            for t_att, t_value in check_atts_ff.items():
                if len(t_value) > 0:
                    for t_key in t_value:
                        t_diff = set(self.players).difference(getattr(self, t_att)[t_key].keys())
                        if len(t_diff) > 0:
                            self.missing_fields.append((t_att, (t_key, t_diff)))
                else:
                    t_diff = set(self.players).difference(getattr(self, t_att).keys())
                    if len(t_diff) > 0:
                        self.missing_fields.append((t_att, t_diff))
        except TypeError:
            pass

    def print(self):
        print(json.dumps(self.__dict__, indent=4))


class Game:
    def __init__(self, raw_data_string, game_number_f=None):
        self.data = raw_data_string
        self.number = game_number_f
        self.hands = self.parse_hands()

        self.players = set()
        self.hand_parse_errors = False
        self.error_hands = None
        self.missing_hands = None
        self.start_hand = None
        self.end_hand = None
        self.total_hands = None
        self.final_outcome = None

        self.combine_game_add = None
        self.combine_player_diff = None

        self.summarize_hands()

    def parse_hands(self):
        hands_f = dict()
        for t_hand in [x for x in self.data.split('\n') if x != '']:
            t_hand_obj = Hand(t_hand)
            hands_f.update({t_hand_obj.number: t_hand_obj})
        return hands_f  # [Hand(x) for x in self.data.split('\n')]

    def get_error_hands(self):
        t_hand_numbers_missing_fields = list()
        self.hand_parse_errors = False  ##### This resets hand_parse_errors, if this flag covers anything besides None key, this reset will lose that info as currently written
        for _, t_hand in self.hands.items():
            if t_hand.number is None:
                self.hand_parse_errors = True
            if len(t_hand.missing_fields) > 0:
                t_hand_numbers_missing_fields.append(t_hand.number)
        return t_hand_numbers_missing_fields

    def check_for_missing_hand_number(self):
        t_f = [int(x) for x in self.hands.keys() if x is not None]
        t_f.sort()
        return [y - 1 for x, y in zip(t_f, t_f[1:]) if y - x != 1]

    def parse_players(self):
        for _, x in self.hands.items():
            if x.number is not None:
                self.players.update(x.players)

    def get_stack_sizes(self):
        if len(self.check_for_missing_hand_number()) > 0:
            print('ERROR:cannot calculate stack size, game missing consecutively numbered hands.')
        else:
            self.hands[str(self.start_hand)].start_stack = dict(
                zip(self.players, [0] * len(self.players)))  # stack at beginning of game (all players set at 0)
            self.hands[str(self.start_hand)].start_stack_rank = dict(
                zip(self.hands[str(self.start_hand)].start_stack.keys(), rankdata([-i for i in self.hands[str(self.start_hand)].start_stack.values()], method='max')))  # account for shifted loop iteration
            for t_h_num in range(int(self.start_hand) + 1, int(self.end_hand) + 1):
                self.hands[str(t_h_num)].start_stack = self.hands[
                    str(t_h_num - 1)].start_stack.copy()  # initialize stack dictionary for hand

                # Check to make sure all player outcomes are accounted for
                if sum(self.hands[str(t_h_num - 1)].outcomes.values()) != 0:
                    print('WARNING: game %s hand %d is not a zero sum outcome hand' % (self.number, t_h_num))

                for t_p, t_s in self.hands[str(t_h_num - 1)].outcomes.items():
                    try:
                        self.hands[str(t_h_num)].start_stack[t_p] = t_s + self.hands[str(t_h_num - 1)].start_stack[
                            t_p]  # add stack at beginning of previous hand + outcome of previous hand
                    except KeyError:
                        pass

                # get rankings of stacks based on relative stack sizes
                self.hands[str(t_h_num)].start_stack_rank = dict(zip(self.hands[str(t_h_num)].start_stack.keys(), rankdata([-i for i in self.hands[str(t_h_num)].start_stack.values()], method='max')))
                
            # add total game outcome to game object
            self.final_outcome = self.hands[str(self.end_hand)].start_stack.copy()
            for t_p, t_s in self.hands[str(self.end_hand)].outcomes.items():
                self.final_outcome.update({t_p: self.final_outcome[t_p] + t_s})
            if sum(self.final_outcome.values()) != 0:
                print('WARNING: Final outcome of game %s is not zero-sum over all players, %f unaccounted for' % (
                self.number, sum(self.final_outcome.values())))

        return None

    def summarize_hands(self):
        self.parse_players()
        self.missing_hands = self.check_for_missing_hand_number()
        self.start_hand = min([int(x) for x in self.hands.keys() if x is not None])
        self.end_hand = max([int(x) for x in self.hands.keys() if x is not None])
        self.total_hands = len(self.hands.keys())
        self.get_stack_sizes()
        self.error_hands = self.get_error_hands()

    def combine_games(self, game2, print_f=True):
        self.combine_game_add = game2.number
        combine_player_set_diff_f = self.players - game2.players
        if len(combine_player_set_diff_f) > 0:
            self.combine_player_diff = combine_player_set_diff_f
            if print_f:
                print("WARNING: Different players for combined games %s and %s" % (self.number, game2.number))
                print("Difference: %s" % self.combine_player_diff)

        self.hands.update(game2.hands)
        self.summarize_hands()

    def drop_bad_hands(self, hand_num_null_TF=True):
        t_num_hands_dropped = 0
        t_hand_numbers = list(
            self.hands.keys())  # structured as such so that dictionary doesn't change size during iteration
        for t_h_num in t_hand_numbers:
            t_pop = False
            if hand_num_null_TF:
                if t_h_num is None:
                    t_pop = True
            # can add more conditions for dropping hand here in same format as hand_num_null_TF

            if t_pop:
                self.hands.pop(t_h_num)
                t_num_hands_dropped += 1

        # summarize output
        if t_num_hands_dropped > 0:
            print("Dropped %d bad hands for game %s" % (t_num_hands_dropped, self.number))
        self.summarize_hands()

    def print(self):
        for t_key, t_value in self.__dict__.items():
            if t_key != 'hands':
                if t_key != 'players':
                    print(json.dumps({t_key: t_value}, indent=4))
                else:
                    print(json.dumps({t_key: list(t_value)}, indent=4))


class Player:
    def __init__(self, name=None):
        self.name = name
        self.game_numbers = None
        self.seat_numbers = None
        self.actions = None
        self.outcomes = None
        self.blinds = None
        self.cards = None
        self.odds = None
        self.stacks = None
        self.stack_ranks = None

    def get_game_numbers(self, games_ff):
        return [x for x in games_ff.keys() if self.name in games_ff[x].players]

    def get_seat_numbers(self, games_ff):
        t_seat_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                try:
                    t_hand_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].players.index(self.name) + 1})
                except:
                    pass
            t_seat_dict.update({t_g_num: t_hand_dict})
        return t_seat_dict

    def get_game_cards(self, games_ff):
        t_card_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                try:
                    t_hand_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].cards['hole_cards'][self.name]})
                except KeyError:
                    pass
            t_card_dict.update({t_g_num: t_hand_dict})
        return t_card_dict

    def get_game_odds(self, games_ff):
        t_odds_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                try:
                    t_hand_dict.update(
                        {str(t_h_num): t_g.hands[str(t_h_num)].odds[self.name]})
                except KeyError:
                    pass
            t_odds_dict.update({t_g_num: t_hand_dict})
        return t_odds_dict

    def get_game_actions(self, games_ff):
        t_action_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                t_round_dict = dict()
                for t_r, t_a in t_g.hands[str(t_h_num)].actions.items():
                    try:
                        t_round_dict.update({t_r: t_a[self.name]})
                    except KeyError:
                        pass
                t_hand_dict.update({str(t_h_num): t_round_dict})
            t_action_dict.update({t_g_num: t_hand_dict})
        return t_action_dict

    def get_game_outcomes(self, games_ff):
        t_outcome_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                try:
                    t_hand_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].outcomes[self.name]})
                except KeyError:
                    pass
            t_outcome_dict.update({t_g_num: t_hand_dict})
        return t_outcome_dict

    def get_blinds(self, games_ff):
        t_blind_dict = dict()
        for t_g_num in self.game_numbers:  # self=p
            t_g = games_ff[t_g_num]
            t_hand_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                t_hand_dict.update({str(t_h_num): {'big': t_g.hands[str(t_h_num)].big_blind == self.name,
                                                   'small': t_g.hands[
                                                                str(t_h_num)].small_blind == self.name}})  # self = p
            t_blind_dict.update({t_g_num: t_hand_dict})
        return t_blind_dict

    def get_stacks(self, games_ff):
        t_stack_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_stack_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                t_hand_stack_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].start_stack[self.name]})
            t_stack_dict.update({t_g_num: t_hand_stack_dict})
        return t_stack_dict

    def get_stack_ranks(self, games_ff):
        t_stack_rank_dict = dict()
        for t_g_num in self.game_numbers:
            t_g = games_ff[t_g_num]
            t_hand_stack_rank_dict = dict()
            for t_h_num in range(t_g.start_hand, t_g.end_hand):
                t_hand_stack_rank_dict.update({str(t_h_num): t_g.hands[str(t_h_num)].start_stack_rank[self.name]})
            t_stack_rank_dict.update({t_g_num: t_hand_stack_rank_dict})
        return t_stack_rank_dict

    def calc_looseness(self, select_hands_ff=None):
        # if no subset is prescribed for calculation use all available player data
        if select_hands_ff is None:
            select_hands_ff = dict()
            for t_g_num in self.game_numbers:
                select_hands_ff.update({t_g_num: list(self.actions[t_g_num].keys())})

        t_hand_count = 0
        t_voluntary_play_count = 0
        for t_g_num, t_h_nums in select_hands_ff.items():
            for t_h_num in t_h_nums:
                t_hand_count += int((not any(self.blinds[t_g_num][t_h_num].values())))
                try:    # if player is eliminated actions dictionary is empty
                    t_voluntary_play_count += int((not any(self.blinds[t_g_num][t_h_num].values())) and
                                                  (self.actions[t_g_num][t_h_num]['preflop'] == 'r' or self.actions[t_g_num][t_h_num]['preflop'] == 'c'))   # count hand if not blind and raised or called, #### pre-flop only configured
                except KeyError:
                    pass
        return t_voluntary_play_count / t_hand_count, t_voluntary_play_count, t_hand_count

    def add_games_info(self, games_f):
        self.game_numbers = self.get_game_numbers(games_f)
        self.seat_numbers = self.get_seat_numbers(games_f)
        self.actions = self.get_game_actions(games_f)
        self.outcomes = self.get_game_outcomes(games_f)
        self.blinds = self.get_blinds(games_f)
        self.cards = self.get_game_cards(games_f)
        self.odds = self.get_game_odds(games_f)
        self.stacks = self.get_stacks(games_f)
        self.stack_ranks = self.get_stack_ranks(games_f)
        self.looseness, _, _ = self.calc_looseness()

    def print(self):
        print(json.dumps(self.__dict__, indent=4))
