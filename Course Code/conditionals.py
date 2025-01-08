edad = input("Ingresa tu edad ") #todo input es texto

edad = int(edad) # "17" -> 17

if edad < 0:
    print("No seas pendejo")
elif edad > 18:
    print("Bienvenido, puedes pasar!")
else:
    print("Acceso denegado")