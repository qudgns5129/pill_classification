import numpy as np
import pandas as pd
import os

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


# 데이터 set 나누기
X_train, X_test, y_train, y_test = train_test_split(X_result, y_result, test_size=0.2, shuffle=True)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# VGG16
vgg_model = VGG16(weights = "imagenet", include_top=True)

x = Dense(35, activation='softmax', name='predictions')(vgg_model.layers[-2].output)

VGG16_model= Model(inputs=vgg_model.input, outputs=x)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
VGG16_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

VGG16_history = VGG16_model.fit(X_train, y_train, batch_size=8, epochs=64, verbose=1)

print('-----------')
print(VGG16_history .history['acc'])
loss_and_metrics = VGG16_model.evaluate(X_test, y_test, batch_size=32)
print('-----------')
print('loss_and_metrics : ' + str(loss_and_metrics))
print('-----------')

CLASSES = np.array(['간장질환용제', '건위소화제', '골격근이완제', '기타의비뇨생식기관및항문용약', '기타의비타민제', '기타의소화기관용약',
       '기타의순환계용약', '기타의알레르기용약', '기타의중추신경용약', '기타의항생물질제제(복합항생물질제제를포함)',
       '기타의혈액및체액용약', '기타의화학요법제', '당뇨병용제', '동맥경화용제', '따로분류되지않는대사성의약품', '무기질제제',
       '비타민A및D제', '소화성궤양용제', '자율신경제', '정신신경용제', '정장제', '제산제',
       '주로그람양성|음성균에작용하는것', '진경제', '진해거담제', '치과구강용약', '칼슘제', '항악성종양제', '항전간제',
       '항히스타민제', '해열.진통.소염제', '혈관확장제', '혈압강하제', '혼합비타민제(비타민AD혼합제제를제외)',
       '효소제제'])

pred = VGG16_model.predict(X_test)
pred_single = CLASSES[np.argmax(pred, axis=-1)]
actual_single = CLASSES[np.argmax(y_test, axis=-1)]

print('-----------')
print('pred_single : ' + str(pred_single[0]))
print('actual_single : ' + str(actual_single[0]))
print('pred_single : ' + str(pred_single[1]))
print('actual_single : ' + str(actual_single[1]))
print('pred_single : ' + str(pred_single[2]))
print('actual_single : ' + str(actual_single[2]))
print('pred_single : ' + str(pred_single[3]))
print('actual_single : ' + str(actual_single[3]))
print('pred_single : ' + str(pred_single[4]))
print('actual_single : ' + str(actual_single[4]))
print('-----------')