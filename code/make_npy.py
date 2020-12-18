import os
import cv2
import pandas as pd
import numpy as np
import random

path = '../seg_image'
names = os.listdir(path)

min_pix = 1
max_pix = 224

for i in range(len(names)):
    img = cv2.imread(path + '\\' + names[i])
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)

    for j in range(0,500):
        rand_w = random.randrange(min_pix, max_pix)
        rand_h = random.randrange(min_pix, max_pix)
        rand_x = random.randrange(min_pix, max_pix)
        rand_y = random.randrange(min_pix, max_pix)

        subimg = img[rand_x:rand_x+rand_h, rand_y:rand_y+rand_w]
        subimg = cv2.resize(subimg, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite('../bbox_image/{}_{}.jpg'.format(names[i][:-4],j), subimg)


# 이미지 불러오기
path = '../bbox_image'
names = os.listdir(path)

img = []
for i in range(len(names)):
    imgs = cv2.imread(path + '\\' + names[i])
    imgs = cv2.resize(imgs, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    img.append(imgs)


# 예측 변수 레이블링
data = pd.read_csv('../data.csv')

labels = []
for i in range(len(names)):
    if names[i].split('_')[2] == '1':
        labels.append('징카민정40mg(은행엽건조엑스)(수출명:징카프란정)')
    elif names[i].split('_')[2] == '2':
        labels.append('아스코푸정(히벤즈산티페피딘)')
    elif names[i].split('_')[2] == '4':
        labels.append('저니스타서방정8밀리그램(히드로모르폰염산염)')
    elif names[i].split('_')[2] == '6':
        labels.append('라니타드정')
    elif names[i].split('_')[2] == '7':
        labels.append('알레기살정10밀리그람(페미로라스트칼륨)')
    elif names[i].split('_')[2] == '9':
        labels.append('로부펜정(록소프로펜나트륨수화물)')
    elif names[i].split('_')[2] == '15':
        labels.append('데놀정(비스무트시트르산염칼륨)')
    elif names[i].split('_')[2] == '16':
        labels.append('드로피진정(레보드로프로피진)')
    elif names[i].split('_')[2] == '17':
        labels.append('우루사정100밀리그램(우르소데옥시콜산)')
    else:
        labels.append('대웅아테놀롤정50밀리그램')


X_result = np.array(img)
y_result = np.array(labels)


np.save('../array/X_result.npy', X_result)
np.save('../array/y_result.npy', y_result)



