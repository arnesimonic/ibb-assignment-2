import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
import matplotlib.pyplot as plt

TRESH = 0.5

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.bounded_path = config['bounded_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self, annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = line.split(" ")[1:5]
                l_arr = [int(i) for i in l_arr]
                annot.append(l_arr)
        return annot

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        class_arr = []
        ps = []
        rs = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        cascade_detector = cascade_detector.Detector()

        import detectors.your_super_detector.super_detector as super_detector
        super_detector = super_detector.SuperDetector()

        positives = 0
        iterated = 0
        for im_name in im_list:

            # Read an image
            img = cv2.imread(im_name)

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)


            # Apply some preprocessing

            # Gamma correction
            # img = preprocess.gamma_correction(img, 1.5)

            # Contrast and brightness
            img = preprocess.contrast_brightness_correction(img, 0.78, 0)

            # Brightness normalization
            # img = preprocess.adjust_brightness(img, 0.6)

            # Automatic brightness
            # img = preprocess.automatic_brightness_and_contrast(img, 15)

            # Kernel
            # img = preprocess.kernel(img)

            # Greyscale
            # img = preprocess.to_greyscale(img)

            # Histogram equalization - This one makes VJ worse
            # img = preprocess.histogram_equlization_rgb(img)

            # CLAHE
            # img = preprocess.clahe(img)

            # Threshold
            # img = preprocess.threshold(img)

            # Adaptive threshold
            # img = preprocess.adaptive_threshold(img)

            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = super_detector.detect(img)
            print(prediction_list)

            # Draw rectangles and save
            im_path = "cropped\\"+im_name
            print(im_path)
            for box in prediction_list:
                [x,y,w,h] = box
                if x<0 or y<0 or w<0 or h<0:
                    continue
                # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
                cropped_img = img[y:y+h, x:x+w]
                cv2.imwrite(im_path, cropped_img)
                break

            # Only for detection:
            p, gt = eval.prepare_for_detection(prediction_list, annot_list)

            iou = eval.iou_compute(p, gt)
            iou_arr.append(iou)

            # Calculate precision and recall
            positive = iou >= TRESH

            class_arr.append(positive)

            iterated += 1
            if positive:
                positives += 1

            p = positives / iterated
            ps.append(p)

            r = positives / len(im_list)
            rs.append(r)

            miou = np.average(iou_arr)

        print("\n")
        print(iou_arr)
        print("Average IOU:", f"{miou}")
        print("\n")

        # cv2.imshow("Slika", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        plt.plot(rs, ps)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
