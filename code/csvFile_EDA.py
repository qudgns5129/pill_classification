
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


raw_data = pd.read_csv('../pill_img/Pillbox.csv')


# In[8]:


is_img = raw_data['has_image'] == True
img_data = raw_data[is_img]


# In[9]:


img_data


# In[16]:


print('Total number of pill names = ' + str(len(img_data['medicine_name'])))
print('Unique number of pill names = ' + str(len(pd.unique(img_data['medicine_name']))))


# - 분류할 class개수는 1979 개이며, 전체 이미지 개수는 8781개
# - 'splimage' 칼럼의 벨류와 이미지의 이름이 매칭되는 것을 확인
# - 'splimprint' 칼럼의 벨류는 약에 프린팅된 텍스트임
