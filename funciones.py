#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grupo: Alta Data
Integrantes: Mariano Papaleo, Gaston Ariel Sanchez, Juan Pablo Aquilante
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def graficar(df,fila):
    plt.imshow(np.array(df.iloc[fila,1:]).reshape(28,28),cmap='Greys')
    numero = df.iloc[fila,0]
    plt.title(f'Numero: {numero}')
    plt.show()
    
def suma_columnas(df):
    suma_columna = []
    a = pd.DataFrame()
    for i in range(len(df.columns)-1):
        suma_columna.append(df.iloc[1:,i].sum())
    a['pixel'] = df.columns
    a = a.drop(0)
    a['suma_de_color'] = suma_columna
    return a

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
    
    plt.plot(valores_k, promedios_train,marker="o",label = 'Train',drawstyle="steps-post")
    plt.plot(valores_k, promedios_test, marker="o",label = 'Test',drawstyle="steps-post")
    plt.legend()
    title = "Accuracy segun profundidad, criterio:" + criterio
    plt.title(title)
    plt.xlabel('Profundidad')
    plt.ylabel('Exactitud (accuracy)')
    archive = "./data/" + nombre_archivo + ".png"
    plt.savefig(archive)
    plt.show()
    
def entrenar_hasta_prof_k(X,Y,criterio,k,nombre_archivo):
    valores_k = range(1,k+1)
    clfs = []
    #Particionamos el conjunto de entrenamiento en 30% test y 70% train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    for d in valores_k:
        clf = DecisionTreeClassifier(criterion = criterio,max_depth = d)
        clf.fit(X_train, Y_train)
        clfs.append(clf)
    #node_counts = [clf.tree_.node_count for clf in clfs]
    #depth = [clf.tree_.max_depth for clf in clfs]
    train_scores = [clf.score(X_train, Y_train) for clf in clfs]
    test_scores = [clf.score(X_test, Y_test) for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("profundidad")
    ax.set_ylabel("accuracy")
    title = "Accuracy segun profundidad, criterio:" + criterio
    ax.set_title(title)
    ax.plot(valores_k, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(valores_k, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    archive = "./data/" + nombre_archivo + ".png"
    plt.savefig(archive)
    plt.show()
    