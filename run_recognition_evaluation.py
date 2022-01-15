import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation
from sklearn.svm import LinearSVC
import math


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.images_path_train = config['images_path_train']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path_train + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)
        print(cla_d)
        
        # Change the following extractors, modify and add your own

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()

        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP()

        import feature_extractors.your_super_extractor.extractor as super_ext
        super = super_ext.Super()
        
        lbp_features_arr = []
        plain_features_arr = []
        super_features_arr = []
        labels = []

        for im_name in im_list:

            # Read an image
            im_name = "/".join(im_name.split("\\"))
            img = cv2.imread(im_name)
            print("Image name: " + im_name)

            # Append the label
            im_id = cla_d['/'.join(im_name.split('/')[-2:])]
            labels.append(im_id)

            # Apply some preprocessing here

            # I have images from last assignment which have already been preprocessed
            # With brightness correction which raised the avg IOU for about 1.5%
            
            # Append the histogram
            lbp_features = lbp.extract(img)
            lbp_features_arr.append(lbp_features)

        print("-----")

        '''
        # Train the model
        model = LinearSVC(C=100.0, random_state=42)
        model.fit(lbp_features_arr, labels)

        all = 0
        correct = 0
        im_list_test = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        for im_name in im_list_test:
            im_name = "/".join(im_name.split("\\"))

            img = cv2.imread(im_name)

            im_id = cla_d['/'.join(im_name.split('/')[-2:])]

            lbp_features = lbp.extract(img)

            prediction = model.predict(lbp_features.reshape(1, -1))[0]

            all += 1
            if prediction == im_id:
                correct += 1
        
        rank1_svm = correct / all * 100
        print('SVM Rank-1[%]', rank1_svm)
        '''

        print(labels)

        # LBP Rank-1 based on distance
        Y_lbp = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')
        r1_lbp = eval.compute_rank1(Y_lbp, labels)
        print('LBP Rank-1[%]', r1_lbp)

    def run_evaluation_awe(self):

        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP()

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()

        eval = Evaluation()

        with open('awe/train.txt') as f:
            lines = f.readlines()
            im_list = lines[0].split(" ")

        labels = []
        lbp_features_arr = []
        p2p_features_arr = []
        for im_num in im_list:

            im_num = int(im_num)

            im_dir = math.ceil(im_num / 10)

            # Append id
            labels.append(im_dir)

            im_dir = str(im_dir)
            while len(im_dir) < 3:
                im_dir = "0" + im_dir

            im_dir_num = int(str(im_num)[-1])

            if im_dir_num == 0:
                im_dir_num = "1" + str(im_dir_num)
            else:
                im_dir_num = "0" + str(im_dir_num)

            im_path = "awe/"+im_dir+"/"+im_dir_num+".png"
            print(im_path)

            # Read Image
            img = cv2.imread(im_path)

            # Extract features
            lbp_features = lbp.extract(img)
            lbp_features_arr.append(lbp_features)

            # Extract features
            p2p_features = pix2pix.extract(img)
            p2p_features_arr.append(p2p_features)

        # LBP Rank-1 based on distance
        Y_lbp = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')
        r1_lbp = eval.compute_rank1(Y_lbp, labels)
        print('LBP Rank-1[%]', r1_lbp)

        # LBP Rank-1 based on distance
        Y_p2p = cdist(p2p_features_arr, p2p_features_arr, 'jensenshannon')
        r1_p2p = eval.compute_rank1(Y_p2p, labels)
        print('P2P Rank-1[%]', r1_p2p)

        # Train the model
        model = LinearSVC(C=100.0, random_state=42)
        model.fit(lbp_features_arr, labels)

        with open('awe/test.txt') as f:
            lines = f.readlines()
            im_list = lines[0].split(" ")

        all = 0
        correct = 0
        for im_num in im_list:

            im_num = int(im_num)

            im_dir = math.ceil(im_num / 10)

            im_id = im_dir

            im_dir = str(im_dir)
            while len(im_dir) < 3:
                im_dir = "0" + im_dir

            im_dir_num = int(str(im_num)[-1])

            if im_dir_num == 0:
                im_dir_num = "1" + str(im_dir_num)
            else:
                im_dir_num = "0" + str(im_dir_num)

            im_path = "awe/"+im_dir+"/"+im_dir_num+".png"
            print(im_path)

            # Read Image
            img = cv2.imread(im_path)

            # Extract features
            lbp_features = lbp.extract(img)

            prediction = model.predict(lbp_features.reshape(1, -1))[0]

            all += 1
            if prediction == im_id:
                correct += 1

        rank1_svm = correct / all * 100
        print('SVM Rank-1[%]', rank1_svm)

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation_awe()
