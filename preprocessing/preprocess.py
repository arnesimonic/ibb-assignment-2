import cv2
import numpy as np

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        # intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        # img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        img = cv2.equalizeHist(img)

        return img

    def to_greyscale(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def clahe(self, img):

        clahe = cv2.createCLAHE(clipLimit=40)
        img = clahe.apply(img)

        return img

    def threshold(self, img):

        th = 80
        max_val = 255
        ret, img = cv2.threshold(img, th, max_val, cv2.THRESH_OTSU)

        return img

    def adaptive_threshold(self, img):
        ret,img = cv2.threshold(img,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return img

    def gamma_correction(self, img, gamma):

        lut = np.empty((1, 256), np.uint8)
        for i in range(256):
            lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        img = cv2.LUT(img, lut)

        return img

    def contrast_brightness_correction(self, img, a, b):

        img = cv2.convertScaleAbs(img, alpha=a, beta=b)

        return img

    def adjust_brightness(self, img, target_brightness):
        cols, rows, channels = img.shape
        brightness = np.sum(img) / (255 * cols * rows)

        ratio = brightness / target_brightness
        if ratio >= 1:
            img = cv2.convertScaleAbs(img, alpha=1/ratio, beta=0)

        return img

    def kernel(self, img):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        return img

    def convertScale(self, img, alpha, beta):
        """Add bias and gain to an image with saturation arithmetics. Unlike
        cv2.convertScaleAbs, it does not take an absolute value, which would lead to
        nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
        becomes 78 with OpenCV, when in fact it should become 0).
        """

        new_img = img * alpha + beta
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        return new_img.astype(np.uint8)

    # Automatic brightness and contrast optimization with optional histogram clipping
    def automatic_brightness_and_contrast(self, image, clip_hist_percent=25):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''

        auto_result = self.convertScale(image, alpha=alpha, beta=beta)
        return auto_result
