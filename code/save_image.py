
# coding: utf-8

# In[7]:


import cv2
import pandas as pd
from urllib import request
import numpy as np


# In[2]:


data = pd.read_csv('c:/users/user/pill/data.csv')


# In[51]:


''' 이미지 저장 완료
urls = np.array(data[['큰제품이미지']]).tolist()

for i in range(len(urls)):
    if urls[i][0] != '-':
        request.urlretrieve(urls[i][0], 'C:/Users/user/pill/pill_img/pill_'+str(i+1)+'.jpg')
        print("{}번, 저장되었습니다.".format(i+1))
    else:
        print('{}번, 저장 실패!'.format(i+1))
'''

