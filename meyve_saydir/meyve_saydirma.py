import cv2
import numpy as np

image = cv2.imread("meyve.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)

# sobel yontemi ile kenar algilama
sobelX = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(blur, cv2.CV_64F, 0, 1)

sobel_combine = cv2.magnitude(sobelX, sobelY)
sobel_combine = np.uint8(sobel_combine)

#görüntüyü binary(siyah-beyaz) hale getir
_, binary_img = cv2.threshold(sobel_combine, 50, 255, cv2.THRESH_BINARY)
#kontur bulma
contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#kontur cizimi
for i, contour in enumerate(contours):
    cv2.drawContours(image, [contour], -1, (0,255,0), 2)
    contour_center = cv2.moments(contour)
    if contour_center["m00"] != 0:
        cX = int(contour_center["m10"] / contour_center["m00"])
        cY = int(contour_center["m01"] / contour_center["m00"])
    else:
        cX, cY = 0, 0

    cv2.putText(image, str(i+1), (cX-10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

cv2.imshow("resim ciktisi", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

















