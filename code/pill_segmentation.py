
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[3]:


# 영상 읽기
img = cv2.imread('../pill_img/pill_1.jpg')
img2 = img.copy()
# 바이너리 이미지로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('imgray', imgray) 

circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
thresh = cv2.threshold(imgray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#cv2.imshow('thresh', thresh) 
erosion = cv2.erode(thresh, circle_kernel)
#cv2.imshow('erosion', erosion)
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, circle_kernel)
#cv2.imshow('opening', opening) 

# 가장 바깥 컨투어만 수집
contour, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 컨투어 갯수와 계층 트리 출력
print(len(contour), hierarchy)

# 가장 바깥 컨투어만 그리기 ---⑤
cv2.drawContours(img, contour, -1, (0,255,0), 5)
# 모든 컨투어 그리기 ---⑥
for idx, cont in enumerate(contour): 
    # 랜덤한 컬러 추출 ---⑦
    color = [int(i) for i in np.random.randint(0,255, 3)]
    # 컨투어 인덱스 마다 랜덤한 색상으로 그리기 ---⑧
    cv2.drawContours(img2, contour, idx, color, 3)
    # 컨투어 첫 좌표에 인덱스 숫자 표시 ---⑨
    cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

# 화면 출력
cv2.imshow('RETR_EXTERNAL', img)
cv2.imshow('RETR_TREE', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[42]:


# 영상 읽기
img = cv2.imread('../pill_img/pill_21.jpg')
img2 = img.copy()
# 바이너리 이미지로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('imgray', imgray) 

thresh = cv2.threshold(imgray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('thresh',thresh)

circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
erosion = cv2.erode(thresh, circle_kernel)
cv2.imshow('erosion', erosion)
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, rect_kernel)
cv2.imshow('opening', opening) 

# 가장 바깥 컨투어만 수집
contour, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 컨투어 갯수와 계층 트리 출력
print(len(contour), hierarchy)

# 가장 바깥 컨투어만 그리기 ---⑤
cv2.drawContours(img, contour, -1, (0,255,0), 5)
# 모든 컨투어 그리기 ---⑥
for idx, cont in enumerate(contour): 
    # 랜덤한 컬러 추출 ---⑦
    color = [int(i) for i in np.random.randint(0,255, 3)]
    # 컨투어 인덱스 마다 랜덤한 색상으로 그리기 ---⑧
    cv2.drawContours(img2, contour, idx, color, 3)
    # 컨투어 첫 좌표에 인덱스 숫자 표시 ---⑨
    cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

# 화면 출력
cv2.imshow('RETR_EXTERNAL', img)
cv2.imshow('RETR_TREE', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

