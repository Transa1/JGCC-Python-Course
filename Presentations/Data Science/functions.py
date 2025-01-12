def sum():
    x = input('Ingrese el primer numero ')
    y = input('Ingrese el segundo numero ')

    result = int(x) + int(y)

    print(result)


def multiplication(x, y):
    print(int(x) * int(y))

def division(a, b):
    return int(a) / int(b)


if __name__ == '__main__':
    sum()
    num1 = input('Ingrese el primer numero ')
    num2 = input('Ingrese el segundo numero ')
    multiplication(num1, num2)
    print(division(10, 5))
    div = division(10, 5)
    print(div + 10)