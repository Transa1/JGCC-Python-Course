def suma():
    x = input('Ingresa el primer numero a sumar ')
    y = input('Ingresa el segundo numero a sumar ')

    resultado = int(x) + int(y)

    print(resultado)

def resta(a,b):
    print(int(a) - int(b))

def multiplicacion():
    w = input('Ingresa el primer factor')
    z = input('Ingresa el segundo factor')
    return int(w), int(z)

def division(x, y):
    return int(x), int(y)

if __name__ == '__main__':
    suma()
    num1 = input('Dame el primer numero a restar')
    num2 = input('Dame el segundo numero a restar')
    resta(num1,num2)
    w,z = multiplicacion()
    print(w * z)
    num3 = input('Dame el primer numero a divir ')
    num4 = input('Dame el segundo numero a divir ')
    a,b = division(num3, num4)
    print(a/b)