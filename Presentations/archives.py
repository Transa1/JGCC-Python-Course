path = 'data.txt'

with open(path, 'w') as file:
    file.write('Hello, World!\n')
    file.write('This is a new line.')

with open(path, 'r') as file:
    for line in file:
        print(line)