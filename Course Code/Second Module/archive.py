path = 'data.txt'

with open(path, 'w') as file:
    file.write("Hola Mundo!!!!!\n")
    file.write("Este es mi otro mensaje")

with open(path, 'r') as file:
    for line in file:
        print(line)