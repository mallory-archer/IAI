from math import ceil


def slansky_strength(hole_cards):

    # Sklansky hand groups: lower group value means better ranking of cards
    # "http://www.thepokerbank.com/strategy/basic/starting-hand-selection/sklansky-groups/"
    hole_cards = hole_cards.upper()

    ranks = "23456789TJQKA"
    suits = "CDHS"

    # input hand for evaluation
    hand = list()
    hand.append(hole_cards[0:2])
    hand.append(hole_cards[2:])

    # sorting the two hand cards in descending order (Expected order : "HigherCard(suit)LowerCard(suit)")
    if ranks.index(hand[0][0]) < ranks.index(hand[1][0]):
        hand[0], hand[1] = hand[1], hand[0]

    # same suit pocket cards groups
    if hand[0][1] == hand[1][1]:
        same_suit_groups = {"AK": 1,
                            "AQ": 2, "AJ": 2, "KQ": 2,
                            "AT": 3, "KJ": 3, "QJ": 3, "JT": 3,
                            "KT": 4, "QT": 4, "J9": 4, "T9": 4, "98": 4,
                            "A9": 5, "A8": 5, "A7": 5, "A6": 5, "A5": 5, "A4": 5, "A3": 5, "A2": 5, "Q9": 5, "T8": 5, "97": 5, "87": 5, "76": 5,
                            "J8": 6, "86": 6, "75": 6, "65": 6, "54": 6,
                            "64": 7, "53": 7, "43": 7, "K9": 7, "K8": 7, "K7": 7, "K6": 7, "K5": 7, "K4": 7, "K3": 7, "K2": 7,
                            "J7": 8, "96": 8, "85": 8, "74": 8, "42": 8, "32": 8
                            }
        # combining the value of cards into single string for dictionary lookup
        hand_value = hand[0][0]+hand[1][0]
        if hand_value in same_suit_groups:
            group = same_suit_groups.get(hand_value)
        else:
            # According to Sklansky the remaining cards fall into group 9
            group = 9

    else:
        # off suit pocket cards groups
        off_suit_groups = {"AA": 1, "KK": 1, "QQ": 1, "JJ": 1,
                           "AK": 2, "TT": 2,
                           "AQ": 3, "99": 3,
                           "AJ": 4, "KQ": 4, "88": 4,
                           "KJ": 5, "QJ": 5, "JT": 5, "77": 5, "66": 5,
                           "AT": 6, "KT": 6, "QT": 6, "55": 6,
                           "J9": 7, "T9": 7, "98": 7, "44": 7, "33": 7, "22": 7,
                           "A9": 8, "K9": 8, "Q9": 8, "J8": 8, "T8": 8, "87": 8, "76": 8, "65": 8, "54": 8
                           }
        # combining the value of cards into single string for dictionary lookup
        hand_value = hand[0][0]+hand[1][0]
        if hand_value in off_suit_groups:
            group = off_suit_groups.get(hand_value)
        else:
            # According to Sklansky the remaining cards fall into group 9
            group = 9

    return group


def chen_strength(hole_cards):

    # Chen formula for assigning strength to pre flop hands, higher is better
    # "http://www.thepokerbank.com/strategy/basic/starting-hand-selection/chen-formula/"
    hole_cards = hole_cards.upper()

    ranks = "23456789TJQKA"
    suits = "CDHS"

    # input hand for evaluation
    hand = list()
    hand.append(hole_cards[0:2])
    hand.append(hole_cards[2:])

    # sorting the two hand cards in descending order (Expected order : "HigherCard(suit)LowerCard(suit)")
    if ranks.index(hand[0][0]) < ranks.index(hand[1][0]):
        hand[0], hand[1] = hand[1], hand[0]

    # Value for Face cards
    facepoints = {"A": 10, "K": 8, "Q": 7, "J": 6, "T": 5}

    a, b = hand[0], hand[1]

    # Score highest card
    if a[0] in facepoints:
        score = facepoints.get(a[0])
    else:
        score = int(a[0])/2.

    # Multiply pairs by 2 of one card's value
    if a[0] is b[0]:
        score *= 2
        if score < 5:
            score = 5

    # Add 2 if cards are suited
    if a[1] is b[1]:
        score += 2

    # Subtract points if there is a gap
    gap = ranks.index(a[0]) - ranks.index(b[0]) - 1
    gapPoints = {1: 1, 2: 2, 3: 4}
    if gap in gapPoints:
        score -= gapPoints.get(gap)
    elif gap >= 4:
        score -= 5

    # Straight bonus
    if (gap < 2) and (ranks.index(a[0]) < ranks.index("Q")) and (a[0] is not b[0]):
        score += 1

    return int(ceil(score))
