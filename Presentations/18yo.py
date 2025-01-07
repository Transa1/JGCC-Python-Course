# Program that asks the user for their age and then tells them if they are old enough to enter a club.

print('Welcome to the club, please enter your age')
age = input('How old are you? ')

if int(age) < 0:
    print('You are not born yet.')
elif int(age) < 18:
    print('You are not old enough to enter this club.')
else:
    print('You are old enough to enter this club.')
    