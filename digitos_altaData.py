# -*- coding: utf-8 -*-

#Carga de archivos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/mnist_desarrollo.csv')

#%%
#Exploracion de datos
def graficar(df,fila):
    plt.imshow(np.array(df.iloc[fila,1:]).reshape(28,28),cmap='Greys')
    numero = df.iloc[fila,0]
    plt.title(f'Numero: {numero}')
    plt.show()
#%%
df_sin_label = np.array(df.iloc[:,1:])
imgs = df_sin_label.reshape(-1,28, 28)

# Calcular la varianza para cada columna
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

label = df.columns[0] #Columna label
con_0s_y_1s = df[ (df[label]==0) | (df[label]==1) ]

#%%
# =============================================================================
# Ejercicio 3
#Para este subconjunto de datos, ver cuántas muestras se tienen y
#determinar si está balanceado entre las clases.
# =============================================================================

#Graficamos una imagen al azar del subconjunto de datos generado
fila = np.random.randint(0, len(con_0s_y_1s))
graficar(con_0s_y_1s,fila)

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
import seaborn as sns
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
X = con_0s_y_1s.iloc[:,[629,630,600]]
Y = con_0s_y_1s[label]

Nrep = 5
#valores_n = [1,3,5,7,10,20]
#valores_n= np.linspace(1,100,20,dtype = int)
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