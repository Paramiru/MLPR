from random import randrange
from random import random
from math import floor
from typing import Dict

def play_game(trials: int) -> Dict:
    results = {'ww': 0, 'bb': 0, 'wb': 0}
    for _ in range(trials):
        # let 0 = black / 1 = white
        upper_face = randrange(0, 2, step=1)
        if (upper_face == 0):
            bottom_face = randrange(0, 2, step=1)
            if (bottom_face == 0):
                results['bb'] += 1
            else:
                results['wb'] += 1
    print(results)

def code_from_answer():
    cards = [[1, 1],
            [0, 0],
            [1, 0]]
    num_cards = len(cards)

    N = 0 # Number of times first side is black
    kk = 0 # Out of those, how many times the other side is white

    for trial in range(int(1e6)):
        card = floor(num_cards * random())
        side = (random() < 0.5)
        other_side = int(not side)
        x1 = cards[card][side]
        x2 = cards[card][other_side]
        if x1 == 0:
            N += 1 # just seen another black face
            kk += (x2 == 1) # count if other side was white

    approx_probability = float(kk) / N
    print(approx_probability)

# play_game(int(1e6))

code_from_answer()