scoreboard = {}

scoreboard['Seahawks'] = 0
scoreboard['Patriots'] = 0
scoreboard['Broncos'] = 0
scoreboard['49ers'] = 0
scoreboard['Raiders'] = 0
scoreboard['Chargers'] = 0

print(scoreboard)

while True:
    print('List of teams:')
    print('1. Seahawks')
    print('2. Patriots')
    print('3. Broncos')
    print('4. 49ers')
    print('5. Raiders')
    print('6. Chargers')
    print('7. Exit')
    choice = input('Enter your choice: ')
    if choice == '7':
        break
    elif choice == '1':
        scoreboard['Seahawks'] += 1
    elif choice == '2':
        scoreboard['Patriots'] += 1
    elif choice == '3':
        scoreboard['Broncos'] += 1
    elif choice == '4':
        scoreboard['49ers'] += 1
    elif choice == '5':
        scoreboard['Raiders'] += 1
    elif choice == '6':
        scoreboard['Chargers'] += 1
    else:
        print('Invalid choice')

    print(scoreboard)