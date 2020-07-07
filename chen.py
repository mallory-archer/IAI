from math import ceil

def pre_flop_strength_2():

    # Chen formula for assigning strength to pre flop hands
    # "http://www.thepokerbank.com/strategy/basic/starting-hand-selection/chen-formula/"

    ranks = "23456789TJQKA"
    suits = "cdhs"

    # input hand for evaluation
    hand = input('Enter the hand:').split()  # ["Ah", "2h"]

    # sorting the two hand cards in descending order (Expected order : "HigherCard(suit)LowerCard(suit)")
    if ranks.index(hand[0][0]) < ranks.index(hand[1][0]):
        hand[0], hand[1] = hand[1], hand[0]

    # Value for Face cards
    facePoints = {"A": 10, "K": 8, "Q": 7, "J": 6, "T": 5}


    a, b = hand[0], hand[1]

    # Score highest card
    if a[0] in facePoints:
        score = facePoints.get(a[0])
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

    print(int(ceil(score)))
    return int(ceil(score))

pre_flop_strength_2()