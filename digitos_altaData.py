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
