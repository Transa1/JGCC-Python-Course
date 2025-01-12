class Auto:
    def __init__(self, marca, modelo, color):
        self.marca = marca
        self.modelo = modelo
        self.color = color
    
    def __str__(self):
        return f'Auto {self.marca} {self.modelo} de color {self.color}'
    
    def encender(self):
        print(f'{self} encendido')

    def apagar(self):
        print(f'{self} apagado')
    
    def acelerar(self):
        print(f'{self} acelerando')

    def frenar(self):
        print(f'{self} frenando')

if __name__ == '__main__':
    auto1 = Auto('Toyota', 'Corolla', 'Blanco')
    auto2 = Auto('Ford', 'Fiesta', 'Rojo')
    auto3 = Auto('Chevrolet', 'Camaro', 'Amarillo')

    print(auto1)
    print(auto2)
    print(auto3)

    auto1.encender()
    auto2.encender()
    auto3.encender()

    auto1.acelerar()
    auto2.acelerar()
    auto3.acelerar()

    auto1.frenar()
    auto2.frenar()
    auto3.frenar()

    auto1.apagar()
    auto2.apagar()
    auto3.apagar()