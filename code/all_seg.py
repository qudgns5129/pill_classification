import cv2
import numpy as np

for idxs in range(1,23224):
    try:
        img = cv2.imread('../original_image/pill_{}.jpg'.format(idxs))
        dst  = cv2.resize(img, dsize=(640, 360), interpolation=cv2.INTER_AREA)
    except:
        continue
        
        
    if data['의약품제형'][idxs-1] == '원형':

        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgray = cv2.medianBlur(imgray, 9)

        rows = imgray.shape[0]
        circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30, minRadius=50, maxRadius=1000)


        if circles is not None:
            circles = np.uint16(np.around(circles))
            for cidx, g in enumerate(circles[0, :]):
                cx, cy = g[0], g[1]
                # circle outline
                radius = g[2]

                x1 = int(cx - radius)
                y1 = int(cy - radius)
                x2 = int(cx + radius)
                y2 = int(cy + radius)
                radius = int(radius)

                crop_img = img[y1:y2, x1:x2, :]
                try:
                    # cv2.imwrite('../seg_image/pillbox_{}_{}.jpg'.format(idxs,cidx), crop_img)
                except:
                    continue
    else:
    
        # 바이너리 이미지로 변환
        imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        erosion = cv2.erode(thresh, circle_kernel)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, rect_kernel) 

        # 컨투어 수집
        contour, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        max_w, max_h = [], []

        for idx, cont in enumerate(contour):
            x, y, w, h = cv2.boundingRect(cont)
            max_w.append((w,h))

            print('width :' + str(w))
            print('height :' + str(h))
        a = sorted(max_w)[-3:]
        print(a)

        for idx, cont in enumerate(contour):
            x, y, w, h = cv2.boundingRect(cont)

            if len(a) == 3:
                if a[2][0] >= 500 and a[2][1] >= 300:
                    del a[2]
                else:
                    del a[0]

            if not (a[0][0] == w and a[0][1] == h or a[1][0] == w and a[1][1] == h): 
                continue

            rect = cv2.minAreaRect(cont)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            subimg = dst[y:y+h, x:x+w]
            # subimg2 = cv2.resize(subimg, dsize=(100, 100), interpolation=cv2.INTER_AREA)
            # cv2.imwrite('../seg_image/pillbox_{}_{}.jpg'.format(idxs,idx), subimg)
        print('{}번째 끝'.format(idxs+1))