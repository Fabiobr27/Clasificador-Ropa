# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:39:04 2020

@author: rubenaguilar98
"""

import tensorflow as tf 
from tensorflow import keras 

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels)=fashion_mnist.load_data()

class_names=['Camiseta','Pantalón','Jersey','Vestido','Abrigo','Sandalia','Camisa','Zapatilla','Bolsa','Botín']
    
train_images = train_images / 255.0

test_images = test_images / 255.0

##plt.figure(figsize=(10,10))
##for i in range(25):
##    plt.subplot(5,5,i+1)
##    plt.xticks([])
##    plt.yticks([])
##    plt.grid(False)
##    plt.imshow(train_images[i], cmap=plt.cm.binary)
##    plt.xlabel(class_names[train_labels[i]]) 
    
#Modelo
    
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Funcion de perdida. Optimizador y Métricas
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Entrenar
history = model.fit(train_images, train_labels, epochs=15, batch_size=500,validation_data=(test_images,test_labels))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,16)

plt.figure()
plt.plot(epochs, loss_values,'b',label='Perdidas de Entrenamiento')
plt.plot(epochs, val_loss_values,'r', label='Perdidas de validacion')
plt.xlabel('Epocas')
plt.ylabel('Perdida')
plt.legend()

plt.show( )
