import numpy as np
import cv2
import matplotlib.pyplot as plt

def Canny_detector(img, weak_th=None, strong_th=None):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.GaussianBlur(img, (5, 5), 1.4)

    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    height, width = img.shape

    mag_max = np.max(mag)
    if weak_th is None:
        weak_th = mag_max * 0.1
    if strong_th is None:
        strong_th = mag_max * 0.5

   
    nms = np.zeros_like(mag)

    for i_x in range(1, width-1):
        for i_y in range(1, height-1):

            grad_ang = ang[i_y, i_x]
            grad_ang = grad_ang % 180

            if (0 <= grad_ang < 22.5) or (157.5 <= grad_ang <= 180):
                before = mag[i_y, i_x - 1]
                after  = mag[i_y, i_x + 1]

            elif (22.5 <= grad_ang < 67.5):
                before = mag[i_y - 1, i_x + 1]
                after  = mag[i_y + 1, i_x - 1]

            elif (67.5 <= grad_ang < 112.5):
                before = mag[i_y - 1, i_x]
                after  = mag[i_y + 1, i_x]

            else:  # 112.5 - 157.5
                before = mag[i_y - 1, i_x - 1]
                after  = mag[i_y + 1, i_x + 1]

            if mag[i_y, i_x] >= before and mag[i_y, i_x] >= after:
                nms[i_y, i_x] = mag[i_y, i_x]
            else:
                nms[i_y, i_x] = 0

    result = np.zeros_like(nms)

    strong = 255
    weak = 75

    for i_x in range(width):
        for i_y in range(height):
            val = nms[i_y, i_x]

            if val >= strong_th:
                result[i_y, i_x] = strong
            elif val >= weak_th:
                result[i_y, i_x] = weak
            else:
                result[i_y, i_x] = 0

    return result

frame = cv2.imread('src/canny_input.png')
if frame is None:
    print("Error: image not found! Please check the path.")

canny_img = Canny_detector(frame)

plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Canny Edges')
plt.imshow(canny_img, cmap='gray')
plt.axis('off')
plt.show()