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
                    cv2.imwrite('../seg_image/pillbox_{}_{}.jpg'.format(idxs,cidx), crop_img)
                except:
                    continue