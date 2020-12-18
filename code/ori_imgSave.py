import cv2
import pandas as pd
from urllib import request
import numpy as np

data = pd.read_csv('../data.csv')

urls = np.array(data[['큰제품이미지']]).tolist()

for i in range(len(urls)):
    if urls[i][0] != '-':
        request.urlretrieve(urls[i][0], '../original_image/pill_'+str(i+1)+'.jpg')
        print("{}번, 저장되었습니다.".format(i+1))
    else:
        print('{}번, 저장 실패!'.format(i+1))