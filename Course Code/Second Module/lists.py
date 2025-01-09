lunas = ['Luna', 'Ceres', 'Deimos', 'Phobos']

print(lunas)

lunas.append('Io')

print(lunas)

lunas.insert(0,'Europa')

print(lunas)

lunas.insert(3,'Ganymede')

print(lunas)

lunas.remove('Phobos')

print(lunas)

del(lunas[3])

print(lunas)

lunas.pop()

print(lunas)

lunas.pop(0)

print(lunas)




#          0        1        2          3
#         -4       -3       -2         -1
lunas = ['Luna', 'Ceres', 'Deimos', 'Phobos']

print(lunas[0])
print(lunas[-1])
print(lunas[-2])

print(lunas[::-1])

print()
print()
print()

numeros = [x for x in range(1,11)]
cuadrados = [x**2 for x in range(1,11)]
potencias = [2**x for x in range(1,13)]

print(numeros)
print(cuadrados)
print(potencias)