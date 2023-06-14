# -*- coding: utf-8 -*-
"""
Grupo: Alta Data
Integrantes: Mariano Papaleo, Gaston Ariel Sanchez, Juan Pablo Aquilante

"""
#Carga de archivos
from funciones import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import time

df = pd.read_csv('./data/mnist_desarrollo.csv',header = None)
df_test = pd.read_csv('./data/mnist_test.csv',header=None)
df_binario_test = pd.read_csv('./data/mnist_test_binario.csv',header = None)

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
df_test = df.rename(columns=dict(zip(df.columns, cols)))
df_binario_test = df.rename(columns=dict(zip(df.columns, cols)))

#%% Exploracion de datos
# vemos la distribucion de los digitos en el dataset

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
#Obtenemos matriz promedio de todo el dataset
df_sin_label = np.array(df.iloc[:,1:])
imgs = df_sin_label.reshape(-1,28, 28)

# Calcular el promedio de cada pixel
matriz_prom = np.mean(imgs, axis=0)

plt.imshow(matriz_prom, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio')
plt.show()

#%%
# =============================================================================
# Ejercicio 2
#Construir un dataframe con el subconjunto que contiene solamente los
#dígitos 0 y 1.
# =============================================================================

df_binario = df[ (df["digito"]==0) | (df["digito"]==1) ]
#Inicializo imagenes promedio para 0s y 1s
con_0s= df[(df["digito"]==0)]
con_1s= df[(df["digito"]==1)]

#Obtenemos imagenes prom para 0 y 1
imgs_con_0 = np.array(con_0s.iloc[:,1:])
ceros = imgs_con_0.reshape(-1,28, 28)
prom_ceros = np.mean(ceros, axis=0)
prom_ceros_dif = prom_ceros

imgs_con_1 = np.array(con_1s.iloc[:,1:])
unos = imgs_con_1.reshape(-1,28, 28)
prom_unos = np.mean(unos, axis=0)
prom_unos_dif = prom_unos

plt.imshow(prom_ceros, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio para el digito 0')
plt.axis('off')
plt.show()

plt.imshow(prom_unos, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio para el digito 1')
plt.axis('off')
plt.show()
#%% Matriz diferencial unos: 
# Sacamos pixeles significativos del 0 de la imagen prom del 1

umbralCeros = 50

for i in range(len(prom_unos_dif)):
    for j in range(len(prom_unos_dif)):
        if(prom_ceros[i][j]>umbralCeros):
            prom_unos_dif[i][j]=0

plt.figure(figsize=(6, 4))
plt.imshow(prom_unos_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen diferencial del 1')
plt.show()
#%% Mas aun, resaltamos pixeles significativos de la imagen resultante
umbralUnos=150

for i in range(0,len(prom_unos_dif)):
    for j in range(0,len(prom_unos_dif[0])):
        if(prom_unos[i][j]<umbralUnos):
            prom_unos_dif[i][j]=0

plt.imshow(prom_unos_dif, cmap='hot')
plt.colorbar()
plt.title('Pixeles representativos del 1')
plt.show()
#busco las columnas relevantes
array_unos_dif=prom_unos_dif.flatten()
pixeles_sign_unos=np.argwhere(array_unos_dif>0)+1

#%% Reseteamos la imagen promedio
imgs_con_0 = np.array(con_0s.iloc[:,1:])
ceros = imgs_con_0.reshape(-1,28, 28)
prom_ceros = np.mean(ceros, axis=0)
prom_ceros_dif = prom_ceros

imgs_con_1 = np.array(con_1s.iloc[:,1:])
unos = imgs_con_1.reshape(-1,28, 28)
prom_unos = np.mean(unos, axis=0)
prom_unos_dif = prom_unos
#%%Matriz diferencial ceros
umbralUnos=50

for i in range(len(prom_ceros_dif)):
    for j in range(len(prom_ceros_dif)):
        if(prom_unos[i][j]>umbralUnos):
            prom_ceros_dif[i][j]=0

plt.figure(figsize=(6, 4))
plt.imshow(prom_ceros_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen diferencial del 0')
plt.show()

#%% 
umbralCeros=150

for i in range(0,len(prom_ceros_dif)):
    for j in range(0,len(prom_ceros_dif[0])):
        if(prom_ceros[i][j]<umbralCeros):
            prom_ceros_dif[i][j]=0

plt.imshow(prom_ceros_dif, cmap='hot')
plt.colorbar()
plt.title('Pixeles representativos del 0')
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
fila = np.random.randint(0, len(df_binario))
graficar(df_binario,fila)
#%% Vemos la distribucion de los digitos en el dataset

# Calcular las ocurrencias de cada dígito
ocurrencias = df_binario['digito'].value_counts()
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
            c="azure",
            fontsize=20)
# Configurar etiquetas y título del gráfico
ax.set_xlabel('Dígitos')
ax.set_ylabel('Ocurrencias')
ax.set_title('Ocurrencias de dígitos')
plt.savefig('./data/Ocurrencias_digitos_binarios.png')
plt.show()
#%%
# =============================================================================
# Ejercicio 4
#Ajustar un modelo de knn considerando pocos atributos, por ejemplo 3.
#Probar con distintos conjuntos de 3 atributos y comparar resultados.
#Analizar utilizando otras cantidades de atributos.
# =============================================================================

# Elegimos 3 atributos(pixeles) aleatoriamente de nuestra matriz de pixeles significativos de 0

filas = pixeles_sign_ceros.shape[0]
filas_aleatorias = np.random.choice(filas, size=3, replace=False)
atributos_aleatorios_ceros = pixeles_sign_ceros[filas_aleatorias] 
print(atributos_aleatorios_ceros)

X = df_binario.iloc[:,np.squeeze(atributos_aleatorios_ceros)]
Y = df_binario.digito

Nrep = 5
valores_n = range(4,21,1)

resultados_test = np.zeros((Nrep, len(valores_n)))
resultados_train = np.zeros((Nrep, len(valores_n)))
cms = []


for i in range(Nrep):
    j=0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    while j < len(valores_n):
        k = valores_n[j]
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        cm = metrics.confusion_matrix(Y_test, Y_pred)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test[i, j] = acc_test
        resultados_train[i, j] = acc_train
        cms.append(cm)
#        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
#                                        display_labels=model.classes_)
#        disp.plot()
#        print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
#        print("Precisión del modelo: ", metrics.precision_score(Y_test, Y_pred, pos_label=1))
#        print("Sensitividad del modelo: ", metrics.recall_score(Y_test, Y_pred, pos_label=1))
#        print("F1 Score del modelo: ", metrics.f1_score(Y_test, Y_pred, pos_label=1))
        j=j+1
#%% Promediamos los resultados
promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test = np.mean(resultados_test, axis = 0) 
#%%

plt.figure(figsize=(7,5),dpi=100)
plt.plot(valores_n, promedios_train, label = 'Train',marker="o",drawstyle="steps-post")
plt.plot(valores_n, promedios_test, label = 'Test',marker="o",drawstyle="steps-post")
plt.legend()
plt.title('Exactitud del modelo de knn con 3 atributos')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')
archive = "./data/knn_k_vecinos_3_atributos_significativos.png"
plt.savefig(archive)
#%% Elegimos 3 atributos aleatoriamente pero de todos los pixeles

#%%
# =============================================================================
# Ejercicio 5
# Para comparar modelos, utilizar validación cruzada. Comparar modelos
# con distintos atributos y con distintos valores de k (vecinos). Para el análisis
# de los resultados, tener en cuenta las medidas de evaluación (por ejemplo,
# la exactitud) y la cantidad de atributos.
# =============================================================================
X = df_binario.iloc[:,[490,462,380]]
Y = df_binario.digito

# CROSS VALIDATION CON KNN Y TREE CON CROSS_VAL_SCORE

clf = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

k_folds = KFold(n_splits = 5)

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
#Sin definir profundidad(sin prepruning)

X = df.iloc[:,1:]
Y = df['digito']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

clf = DecisionTreeClassifier(criterion = "entropy")
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
Y_pred_train = clf.predict(X_train)
acc_test = metrics.accuracy_score(Y_test, Y_pred)
acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
print("Criterio: entropy")
print("Test:",acc_test)
print("Train:",acc_train)
print("Profundidad:",clf.tree_.max_depth) #20
print()
clf = DecisionTreeClassifier(criterion = "gini")
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
Y_pred_train = clf.predict(X_train)
acc_test = metrics.accuracy_score(Y_test, Y_pred)
acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
print("Criterio: gini")
print("Test:",acc_test)
print("Train:",acc_train)
print("Profundidad:",clf.tree_.max_depth) #47
#%%

# Iniciar el contador de tiempo
start_time = time.time()

entrenar_y_graficar(X, Y, "entropy",5, 20, "entropy_k_20_n_5reps_")
entrenar_y_graficar(X, Y, "gini",5, 20, "entropy_k_20_5reps_")      

end_time = time.time()
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time} segundos") #21 minutos de ejecucion
#%% Analizamos distintas profundidades

X = df.iloc[:,1:]
Y = df['digito']

# Iniciar el contador de tiempo
start_time = time.time()
# Código que deseas medir
entrenar_hasta_prof_k(X,Y,"gini",20,"clf_hasta_20k_gini")
entrenar_hasta_prof_k(X,Y,"entropy",20,"clf_hasta_20k_entropy") 
# Finalizar el contador de tiempo y calcular la duración
end_time = time.time()
execution_time = end_time - start_time

print(f"Tiempo de ejecución: {execution_time} segundos")#5 minutos de ejecucion
#%%
#Al parecer 12 es la profundidad optima, veamos por criterio
clf = DecisionTreeClassifier(criterion = "entropy",max_depth = 12)
k_folds = KFold(n_splits = 10)
scores_Entropy = cross_val_score(clf, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores_Entropy)
print("Average CV Score: ", scores_Entropy.mean())
print("Number of CV Scores used in Average: ", len(scores_Entropy))

clf = DecisionTreeClassifier(criterion = "gini",max_depth = 12)
k_folds = KFold(n_splits = 10)
scores_Gini = cross_val_score(clf, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores_Gini)
print("Average CV Score: ", scores_Gini.mean())
print("Number of CV Scores used in Average: ", len(scores_Gini))

mejor_criterio = "Entropy" if scores_Entropy.mean() > scores_Gini.mean() else "Gini"
print("Mejor criterio:",mejor_criterio)#Entropy
#%%
# =============================================================================
# Ejercicio 7
# Para comparar y seleccionar los árboles de decisión, utilizar validación
# cruzada con k-folding.
# =============================================================================

clfs=[]
profundidades = [i for i in range(5,20)]
k_folds = KFold(n_splits = 5)
for k in profundidades:
	clf = DecisionTreeClassifier(criterion = "entropy",max_depth = k)
	clfs.append(clf)
scores = [cross_val_score(clf, X, Y, cv = k_folds).mean() for clf in clfs]
max_score = np.argmax(scores)
prof_optima = profundidades[max_score]
print("profundida optima: ",prof_optima) #14

#%%
clf = DecisionTreeClassifier(criterion = "entropy",max_depth = 14)
knn = KNeighborsClassifier(n_neighbors=12)

X = df.iloc[:,1:]
Y = df['digito']
k_folds = KFold(n_splits = 10)

scores_knn = cross_val_score(knn, X, Y, cv = k_folds)
print("KNN-k:12")
print("Cross Validation Scores: ", scores_knn)
print("Average CV Score: ", scores_knn.mean())
print("Number of CV Scores used in Average: ", len(scores_knn)) #5 min de ejecucion

scores_clf = cross_val_score(clf, X, Y, cv = k_folds)

print("Decision tree-profundidad:12")
print("Cross Validation Scores: ", scores_clf)
print("Average CV Score: ", scores_clf.mean())
print("Number of CV Scores used in Average: ", len(scores_clf))

mejor_modelo = "clf max_depth=12" if scores_clf.mean() > scores_knn.mean() else "KNN k=12"
print("mejor modelo: ",mejor_modelo)
#%%
""" PREDICCIÓN DF_TEST """

X = df.iloc[:,1:]
Y = df['digito']
clf = DecisionTreeClassifier(criterion = "entropy",max_depth=14)
clf.fit(X, Y)
Y_pred_train = clf.predict(X)
acc_train = metrics.accuracy_score(Y, Y_pred_train)
print("Predicción de digitos en el df_test con DecisionTreeClassifier, Criterio: entropy")
print("Train:",acc_train)#0.991

X_test = df_test.iloc[:,1:]
Y_test = df_test['digito']

Y_pred = clf.predict(X_test)
acc_test = metrics.accuracy_score(Y_test, Y_pred)

print("Test:",acc_test)#0.8871
#%%
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, Y)
Y_pred_train = knn.predict(X)
acc_train = metrics.accuracy_score(Y, Y_pred_train)
print("Predicción de digitos en el df_test con KNN, k=12")
print("Train:",acc_train)#0.97
Y_pred = knn.predict(X_test)
acc_test = metrics.accuracy_score(Y_test, Y_pred)

print("Test:",acc_test)#0.97