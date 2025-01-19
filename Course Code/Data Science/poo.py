class Auto:
    def __init__(self, marca, modelo, color):
        self.marca = marca
        self.modelo = modelo
        self.color = color
    
    def __str__(self):
        return f'Auto {self.marca} {self.modelo} de color {self.color}'
    
    def encender(self):
        print(f'{self} encendido')

    def acelerar(self):
        print(f'{self} acelerando')

    def frenar(self):
        print(f'{self} frenando')

    def apagar(self):
        print(f'{self} apagado')

if __name__ == '__main__':
    corolla = Auto('Toyota', 'Corolla', 'Rojo')
    camaro = Auto('Chevrolet', 'Camaro', 'Amarillo')

    print(corolla)
    corolla.encender()
    corolla.acelerar()
    corolla.frenar()
    corolla.apagar()

    print()
    print(camaro)
    camaro.encender()
    camaro.acelerar()
    camaro.frenar()
    camaro.apagar()