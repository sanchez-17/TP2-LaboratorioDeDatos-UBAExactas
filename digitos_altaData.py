# -*- coding: utf-8 -*-

#Carga de archivos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv('./data/mnist_desarrollo.csv')

# =============================================================================
#Ejercicio 1
#Realizar un análisis exploratorio de los datos. Ver, entre otras cosas,
#cantidad de datos, cantidad y tipos de atributos, cantidad de clases de la
#variable de interés (el dígito) y otras características que consideren
#relevantes. ¿Cuáles parecen ser atributos relevantes? ¿Cuáles no? Se
#pueden hacer gráficos para abordar estas preguntas.
# =============================================================================

#La primera columna indica el digito, las demas son los pixeles de la imagen.renombramos columnas:
#Cada pixel será representado de la forma i-j: indicando fila y columna
cols = ["digito"]
for i in range(28):
    for j in range(28):
        elem = str(i) + "-" + str(j)
        cols.append(elem)


df = df.rename(columns=dict(zip(df.columns, cols)))
#%%

#Exploracion de datos
def graficar(df,fila):
    plt.imshow(np.array(df.iloc[fila,1:]).reshape(28,28),cmap='Greys')
    numero = df.iloc[fila,0]
    plt.title(f'Numero: {numero}')
    plt.show()

#Las proporciones de los dígitos en todo el dataset
cant_de_imgs_por_num = df["digito"].value_counts().sort_index()
porc_de_imgs_por_num = round(cant_de_imgs_por_num / len(df) * 100,2)
proporcion_digitos = pd.DataFrame({'cant': cant_de_imgs_por_num,'% cant.':porc_de_imgs_por_num})
proporcion_digitos.index.name = 'Dígito'
proporcion_digitos = proporcion_digitos.sort_values(by='cant')
#%%

# Histograma para proporcion de digitos
plt.hist(df["digito"], orientation = "vertical")

#%%
tabla = pd.plotting.table(plt.figure(), proporcion_digitos, loc='center')

# Establecer el estilo de la tabla
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)
tabla.scale(1.2, 1.2)

# Ocultar los ejes del gráfico
plt.axis('off')

# Exportar el gráfico de tabla como una imagen
plt.savefig('tabla.png', bbox_inches='tight')
#%%
df_sin_label = np.array(df.iloc[:,1:])
imgs = df_sin_label.reshape(-1,28, 28)

# Calcular el promedio de cada pixel
matriz_prom = np.mean(imgs, axis=0)

plt.imshow(matriz_prom, cmap='hot')
plt.colorbar()
plt.title('Mapa de calor de dispersión')
plt.show()

#%%
# =============================================================================
# Ejercicio 2
#Construir un dataframe con el subconjunto que contiene solamente los
#dígitos 0 y 1.
# =============================================================================

con_0s_y_1s = df[ (df["digito"]==0) | (df["digito"]==1) ]

#%%
# =============================================================================
# Ejercicio 3
#Para este subconjunto de datos, ver cuántas muestras se tienen y
#determinar si está balanceado entre las clases.
# =============================================================================

#Graficamos una imagen al azar del subconjunto de datos generado
fila = np.random.randint(0, len(con_0s_y_1s))
graficar(con_0s_y_1s,fila)

# Esto calcula para cada pixel la suma total de los valores que tienen a lo largo de todas las imagenes
# Arma un dataframe cuya primera columna es el pixel, y la segunda es el valor total sumado

def suma_columnas(df):
    suma_columna = []
    a = pd.DataFrame()
    for i in range(len(df.columns)-1):
        suma_columna.append(df.iloc[1:,i].sum())
    a['pixel'] = df.columns
    a = a.drop(0)
    a['suma_de_color'] = suma_columna
    return a

columnas = suma_columnas(df)
columnas_ceros_y_unos = suma_columnas(con_0s_y_1s)

# Dataframe de pixeles que tienen unicamente el valor 0 a lo largo de todas las imagenes
sub = columnas[columnas['suma_de_color'] == 0]
print("Proporcion de pixeles que tienen unicamente el valor 0 a lo largo de todas las imagenes del dataset")
len(sub)/(len(df.columns)-1) * 100 # 66/784*100

#Vemos cuantas muestras se tienen
cant_de_imgs_por_num = con_0s_y_1s[label].value_counts().sort_index()
porc_de_imgs_por_num = round(cant_de_imgs_por_num / len(con_0s_y_1s) * 100,2)
cant = pd.DataFrame({'cantidad': cant_de_imgs_por_num})
porcentajes = pd.DataFrame({'% subconj':porc_de_imgs_por_num,'% dataset original':round(cant_de_imgs_por_num / len(df) * 100,2)})
cant.index.name = 'Dígito'
tabla = pd.concat([cant, porcentajes], axis=1)

#%%
# =============================================================================
# Ejercicio 4
#Ajustar un modelo de knn considerando pocos atributos, por ejemplo 3.
#Probar con distintos conjuntos de 3 atributos y comparar resultados.
#Analizar utilizando otras cantidades de atributos.
# =============================================================================
#%%

#X = df_0.iloc[:,[629,630,600]]
#Y = df_0.digito
# Elegimos 3 atributos(pixeles)
X = con_0s_y_1s.iloc[:,[213,214,241]]
Y = con_0s_y_1s.digito


Nrep = 5
valores_n = [5,10,15,20]

resultados_test = np.zeros((Nrep, len(valores_n)))
resultados_train = np.zeros((Nrep, len(valores_n)))


for i in range(Nrep):
    j=0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    while j < len(valores_n):
        k = valores_n[j]
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test[i, j] = acc_test
        resultados_train[i, j] = acc_train
        j=j+1

#%%

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test = np.mean(resultados_test, axis = 0) 
#%%

plt.figure(figsize=(7,5),dpi=100)
plt.plot(valores_n, promedios_train, label = 'Train')
plt.plot(valores_n, promedios_test, label = 'Test')
plt.legend()
plt.title('Exactitud del modelo de knn')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')

# =============================================================================
# Ejercicio 5
# Para comparar modelos, utilizar validación cruzada. Comparar modelos
# con distintos atributos y con distintos valores de k (vecinos). Para el análisis
# de los resultados, tener en cuenta las medidas de evaluación (por ejemplo,
# la exactitud) y la cantidad de atributos.
# =============================================================================
