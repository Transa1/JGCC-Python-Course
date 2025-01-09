scoreboard = {}

scoreboard['Seahawks'] = 0
scoreboard['Patriots'] = 0
scoreboard['Broncos'] = 0
scoreboard['49ers'] = 0
scoreboard['Raiders'] = 0
scoreboard['Chargers'] = 0

print(scoreboard)

while True:
    print('Lista de Equipos')
    print('1.- Seahawks')
    print('2.- Patriots')
    print('3.- Broncos')
    print('4.- 49ers')
    print('5.- Raiders')
    print('6.- Chargers')
    print('7.- Salir')

    opcion = input('Ingresa el equipo ganador, presione 7 para salir')

    if opcion == '7':
        break
    elif opcion == '1':
        scoreboard['Seahawks'] += 1
    elif opcion == '2':
        scoreboard['Patriots'] += 1
    elif opcion == '3':
        scoreboard['Broncos'] += 1
    elif opcion == '4':
        scoreboard['49ers'] += 1
    elif opcion == '5':
        scoreboard['Raiders'] += 1
    elif opcion == '6':
        scoreboard['Chargers'] += 1
    else:
        print('Opcion Invalida')
    
    print(scoreboard)