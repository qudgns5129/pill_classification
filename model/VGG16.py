import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras import backend as K

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Dropout, MaxPooling2D, Activation,
    Flatten, Dense, Input, Concatenate, LeakyReLU, Add, AveragePooling2D, ReLU, MaxPool2D, ZeroPadding2D,
    AveragePooling2D, Reshape
)
from tensorflow.keras import initializers, regularizers, metrics

from tensorflow.keras.applications.vgg16 import VGG16

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# F1-score metric
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


X_result = np.load('X_result.npy')
y_result = np.load('y_result.npy')

# 데이터 set 나누기
X_train, X_test, y_train, y_test = train_test_split(X_result, y_result, test_size=0.2, shuffle=True)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# VGG16
vgg_model = VGG16(weights = "imagenet", include_top=False, input_shape=(224,224,3))
"""
for layer in vgg_model.layers[:19]:
    layer.trainable = False
 """
out = vgg_model.output
x = Flatten()(out)
x = Dense(units=4096, activation='relu')(x)
x = Dense(units=4096, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

VGG16_model = Model(vgg_model.input, outputs=outputs)

optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.001)
VGG16_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', tensorflow.keras.metrics.Precision(name='precision'), tensorflow.keras.metrics.Recall(name='recall'), f1_m])

callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)]

VGG16_history = VGG16_model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=64, verbose=1, callbacks=callbacks)

print('-----------')
loss_and_metrics = VGG16_model.evaluate(X_test, y_test, batch_size=32)
print('-----------')
print('loss_and_metrics : ' + str(loss_and_metrics))
print('-----------')

y_vloss = VGG16_history.history['val_loss']
y_loss = VGG16_history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.savefig('VGG16_plot.png')
