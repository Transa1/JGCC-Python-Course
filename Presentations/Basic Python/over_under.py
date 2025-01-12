# over under game

import random

random_number = random.randint(1, 100)

while True:
    guess = int(input('Enter a number between 1 and 100: '))
    if guess < random_number:
        print('Too low!')
    elif guess > random_number:
        print('Too high!')
    else:
        print('You win!')
        break