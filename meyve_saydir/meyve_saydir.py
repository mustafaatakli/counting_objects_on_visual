import cv2
import numpy as np

image = "meyve.jpeg"
class goruntuIsleme:
    def __init__(self, image):
        self.image_path = image
        self.image = None
        self.gray_img = None
        self.blur_img = None
        self.sobel_X = None
        self.sobel_Y = None
        self.sobel_combine = None
        self.binary_img = None
        self.contours = None

    def load_img(self):
        self.image = cv2.imread(self.image_path)
    def convert_to_gray(self):
        self.gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    def gaussian_blur(self, kernel_size=(5, 5)):
        self.blur_img = cv2.GaussianBlur(self.gray_img, kernel_size, 0)

    #sobel yontemi ile kenar algilama
    def sobel_method(self):
        self.sobel_X = cv2.Sobel(self.blur_img, cv2.CV_64F, 1, 0)
        self.sobel_Y = cv2.Sobel(self.blur_img, cv2.CV_64F, 0, 1)
        self.sobel_combine = cv2.magnitude(self.sobel_X, self.sobel_Y)
        self.sobel_combine = np.uint8(self.sobel_combine)

    #görüntüyü binary(siyah-beyaz) hale getir
    def convert_to_binary(self, threshold=50, max_value=255):
        _, self.binary_img = cv2.threshold(self.sobel_combine, threshold, max_value, cv2.THRESH_BINARY)

    #kontur bulma
    def contour_find(self):
        self.contours, _ = cv2.findContours(self.binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #kontur cizimi
    def contour_draw(self):
        for i, contour in enumerate(self.contours):
            cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 2)
            contour_center = cv2.moments(contour)
            if contour_center["m00"] != 0:
                cX = int(contour_center["m10"] / contour_center["m00"])
                cY = int(contour_center["m01"] / contour_center["m00"])
            else:
                cX, cY = 0, 0
            cv2.putText(self.image, str(i + 1), (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    def show_img(self, window_name="resim-ciktisi"):
        cv2.imshow(window_name, self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

output = goruntuIsleme(image)
output.load_img()
output.convert_to_gray()
output.gaussian_blur()
output.sobel_method()
output.convert_to_binary()
output.contour_find()
output.contour_draw()
output.show_img()