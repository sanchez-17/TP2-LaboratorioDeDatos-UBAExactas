# -*- coding: utf-8 -*-

#Carga de archivos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
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
# Calcular las ocurrencias de cada dígito
ocurrencias = df['digito'].value_counts()
# Calcular el total de ocurrencias
total_ocurrencias = ocurrencias.sum()
# Calcular los porcentajes de ocurrencia
porcentajes = (ocurrencias / total_ocurrencias) * 100
# Crear la figura y el eje del gráfico
fig, ax = plt.subplots()
digitos = [str(d) for d in ocurrencias.index]
plt.bar(digitos,ocurrencias.values)
#Agregar etiquetas de texto en cada barra
for i in range(len(ocurrencias)):
    ax.text(i , ocurrencias.values[i],
            f"{porcentajes.values[i]:.2f}%",
            ha='center',
            va='top',
            rotation=60,
            c="azure")
# Configurar etiquetas y título del gráfico
ax.set_xlabel('Dígitos')
ax.set_ylabel('Ocurrencias')
ax.set_title('Ocurrencias de dígitos')
plt.savefig('./data/Ocurrencias_digitos.png')
plt.show()
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
#Inicializo imagenes promedio para 0s y 1s
con_0s= df[(df["digito"]==0)]
con_1s= df[(df["digito"]==1)]

imgs_con_0 = np.array(con_0s.iloc[:,1:])
ceros = imgs_con_0.reshape(-1,28, 28)
prom_ceros = np.mean(ceros, axis=0)
prom_ceros_dif = prom_ceros
umbralUnos=200

imgs_con_1 = np.array(con_1s.iloc[:,1:])
unos = imgs_con_1.reshape(-1,28, 28)
prom_uno = np.mean(unos, axis=0)
prom_uno_dif = prom_uno
umbralCeros=150

plt.imshow(prom_ceros, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (0)')
plt.show()

plt.imshow(prom_uno, cmap='hot')
plt.colorbar()
plt.title('Imagen  (1)')
plt.show()
#%% Matriz diferencial unos

for i in range(len(prom_uno_dif)):
    for j in range(len(prom_uno_dif)):
        if(prom_cpixeles_sign_unoeros[i][j]>umbralCeros):
            prom_uno_dif[i][j]=0


plt.figure(figsize=(6, 4))
plt.imshow(prom_uno_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (1) sin representativos del 0')
plt.show()

umbralUnos=200

for i in range(0,len(prom_uno_dif)):
    for j in range(0,len(prom_uno_dif[0])):
        if(prom_uno[i][j]<umbralUnos):
            prom_uno_dif[i][j]=0

plt.imshow(prom_uno_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (1) diferencial')
plt.show()
#busco las columnas relevantes
array_unos_dif=prom_uno_dif.flatten()
pixeles_sign_uno=np.argwhere(array_unos_dif>0)+1

#%% Matriz diferencial ceros

umbralUnos=150

for i in range(len(prom_ceros_dif)):
    for j in range(len(prom_ceros_dif)):
        if(prom_uno[i][j]>umbralUnos):
            prom_ceros_dif[i][j]=0

plt.figure(figsize=(6, 4))
plt.imshow(prom_ceros_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (0) sin representativos del 1')
plt.show()

umbralCeros=175

for i in range(0,len(prom_ceros_dif)):
    for j in range(0,len(prom_ceros_dif[0])):
        if(prom_ceros[i][j]<umbralCeros):
            prom_ceros_dif[i][j]=0

plt.imshow(prom_ceros_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (0) diferencial')
plt.show()

array_ceros_dif=prom_ceros_dif.flatten()
pixeles_sign_ceros=np.argwhere(array_ceros_dif>0)+1            

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
X = con_0s_y_1s.iloc[:,[490,462,380]]
Y = con_0s_y_1s.digito

# CROSS VALIDATION CON KNN Y TREE CON CROSS_VAL_SCORE

clf = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(clf, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

scores = cross_val_score(knn, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

#%%

# CROSS VALIDATION CON KNN CON CROSS_VALIDATE

cv_results = cross_validate(knn, X, Y, cv=10,return_train_score=True)
sorted(cv_results.keys())
print('Test score:', cv_results['test_score'])
print('Train score:', cv_results['train_score'])
print('Promedio Test score: ', np.mean(cv_results['test_score']))
print('Promedio Train score: ', np.mean(cv_results['train_score']))

#%%

# CROSS VALIDATION CON KFOLD.SPLIT (FALLO)

kf = KFold(n_splits=5)

# Inicializar una lista para almacenar los resultados de precisión
accuracy_scores = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # Inicializar y ajustar el clasificador KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    
    # Realizar predicciones en el conjunto de prueba
    Y_pred = knn.predict(X_test)
    
    # Calcular la precisión y agregarla a la lista de resultados
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_scores.append(accuracy)

# Calcular el promedio de los resultados de precisión
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

# Imprimir el resultado final
print("Precisión promedio:", average_accuracy)
#%%
# =============================================================================
# Ejercicio 6
# Trabajar nuevamente con el dataset de todos los dígitos. Ajustar un
# modelo de árbol de decisión. Analizar distintas profundidades.
# =============================================================================

def entrenar_y_graficar(X,Y,criterio,Nrep,k,nombre_archivo):
    valores_k = range(1,k+1)
    resultados_test = np.zeros( (Nrep,k))
    resultados_train = np.zeros( (Nrep,k))

    for i in range(Nrep):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
        for j in valores_k:
            model = DecisionTreeClassifier(criterion = criterio,max_depth = j)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            Y_pred_train = model.predict(X_train)
            acc_test = metrics.accuracy_score(Y_test, Y_pred)
            acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
            resultados_test[i,j-1] = acc_test
            resultados_train[i,j-1] = acc_train
    
    promedios_train = np.mean(resultados_train, axis = 0) #A lo largo de cada columna
    promedios_test = np.mean(resultados_test, axis = 0)
    
    plt.plot(valores_k, promedios_train, label = 'Train')
    plt.plot(valores_k, promedios_test, label = 'Test')
    plt.legend()
    plt.title('Exactitud de arboles de decision')
    plt.xlabel('Profundidad')
    plt.ylabel('Exactitud (accuracy)')
    archive = "./data/" + nombre_archivo + ".png"
    plt.savefig(archive)
    plt.show()

X = df.iloc[:,1:]
Y = df['digito']
entrenar_y_graficar(X,Y,"entropy",2,20,"entropy_k_5_N_3")
