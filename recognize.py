# import the necessary packages
from extract_feature.lbp import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support as score

import argparse
import cv2
import os
import random

class recognize:
    def __init__(self, image_dir, gt_dir, lbn_points, lbn_radius):
        self.IMAGE_DIR = image_dir
        self.GT_DIR = gt_dir
        self.TRAIN_IMAGES = []
        self.TEST_IMAGES = []
        self.TRAIN_LABELS = []
        self.TEST_LABELS = []
        self.TEST_NUM = 0
        self.PRED_LABELS = []

        # FEATURE MODEL
        self.desc = LocalBinaryPatterns(int(lbn_points), int(lbn_radius))
        # SVM model
        self.model = LinearSVC(C=100.0, random_state=42)

    def prepare(self):
        print("Data DIR:", os.path.join(self.IMAGE_DIR))
        print("GroundTruth DIR:", os.path.join(self.GT_DIR))
        images_list = next(os.walk(os.path.join(self.IMAGE_DIR)))[1]
        print("Image List: ", len(images_list))
        self.TEST_IMAGES = random.sample(images_list, self.TEST_NUM)
        self.TRAIN_IMAGES = list(set(images_list) - set(self.TEST_IMAGES))


    def train(self):
        train_labels = []
        lbn_data = []
        for case in self.TRAIN_IMAGES:
            samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
            for sample in samples:
                imagePath = os.path.join(self.IMAGE_DIR, case, os.path.splitext(sample)[0] + ".tif")
                image = cv2.imread(imagePath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = self.desc.describe(gray)
                label_path = os.path.join(self.GT_DIR, case, os.path.splitext(sample)[0] + ".csv")
                lable = 1 if os.path.isfile(label_path) else 0
                train_labels.append(lable)
                lbn_data.append(hist)
        # train a Linear SVM on the data
        self.model.fit(lbn_data, train_labels)

    def test(self):
        # loop over the testing images
        for case in self.TEST_IMAGES:
            samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
            for sample in samples:
                imagePath = os.path.join(self.IMAGE_DIR, case, os.path.splitext(sample)[0] + ".tif")
                image = cv2.imread(imagePath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = self.desc.describe(gray)
                label_path = os.path.join(self.GT_DIR, case, os.path.splitext(sample)[0] + ".csv")
                label = 1 if os.path.isfile(label_path) else 0
                prediction = self.model.predict(hist.reshape(1, -1))
                self.TEST_LABELS.append(label)
                self.PRED_LABELS.append(prediction)

    def eval(self):
        precision, recall, fscore, support = score(self.TEST_LABELS, self.PRED_LABELS)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="path to the training images")
    ap.add_argument("-gt", "--gt", required=True,
                    help="path to the tesitng images")
    ap.add_argument("-lbn_points", "--lbn_points", required=True,
                    help="LBN points param")
    ap.add_argument("-lbn_rad", "--lbn_radius", required=True,
                    help="LBN radius param")
    args = vars(ap.parse_args())



    mitoses_model = recognize(args["data"],args["gt"],args["lbn_points"],args["lbn_radius"])
    mitoses_model.prepare()
    mitoses_model.train()
    mitoses_model.test()
    mitoses_model.eval()








