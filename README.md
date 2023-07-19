#  TP2 de Laboratorio de Datos: Reconocimiento de digitos, clasificación y validación cruzada

## Enunciado: Clasificación y Validación Cruzada con el dataset MNIST
En este repositorio se presenta el desarrollo del Trabajo Práctico de la materia Laboratorio de Datos (1C2023) de la Facultad de Ciencias Exactas y Naturales (FCEyN) de la Universidad de Buenos Aires (UBA). El trabajo se enfoca en el análisis de clasificación y selección de modelos utilizando validación cruzada con el conjunto de datos MNIST, el cual consiste en un conjunto de imágenes que representan dígitos escritos a mano.

Se exploraron distintos modelos de clasificación, como K-Nearest Neighbors (KNN) y Árboles de Decisión, y se aplicaron técnicas de validación cruzada para comparar y seleccionar los mejores modelos. También se realizó un análisis exploratorio de los datos para identificar características relevantes.

## Equipo

- Gaston Ariel Sanchez      (gasanchez@dc.uba.ar)
- Mariano Papaleo           (gagopoliscool@gmail.com)
- Juan Pablo Hugo Aquilante (aquilantejp@outlook.es)

## Código y Visualizaciones
En este repositorio se encuentran los códigos implementados para cada paso del enunciado, así como también las visualizaciones y análisis de resultados obtenidos durante el desarrollo del trabajo.

Para más detalles sobre el análisis y los resultados, consultar el código y las visualizaciones en este repositorio.

### Descomprimir los datos de entrenamiento(desarrollo)

```
gunzip -c data/mnist_desarrollo.csv.gz > data/mnist_desarrollo.csv
```

Luego los datos de test
```
gunzip -c data/mnist_test.csv.gz > data/mnist_test.csv
gunzip -c data/mnist_test_binario.csv.gz > data/mnist_test_binario.csv
```
