# import the necessary packages
from extract_feature.lbp import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm
import argparse
import cv2
import os
import random
from sklearn.externals import joblib
import time


class recognize:
    def __init__(self, image_dir, gt_dir):
        self.IMAGE_DIR = image_dir
        self.GT_DIR = gt_dir
        self.TRAIN_IMAGES = []
        self.TEST_IMAGES = []
        self.TRAIN_LABELS = []
        self.TEST_LABELS = []
        self.TEST_NUM = 10
        self.PRED_LABELS = []
        self.TIMESTR = time.strftime("%Y%m%d-%H%M%S")

        # FEATURE MODEL
        self.LBP_points = 8
        self.LBP_radius = 1
        self.desc = LocalBinaryPatterns(self.LBP_points, self.LBP_radius)

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
        print("******* Training ********")
        train_labels = []
        lbn_data = []
        for case in tqdm(self.TRAIN_IMAGES):
            samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
            for sample in tqdm(samples, desc=case):
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
        # export trained model
        joblib.dump(self.model, 'logs/lbp_svm_'+self.TIMESTR+'.pkl', compress=9)


    def test(self):
        print("******* Testing ********")
        # loop over the testing images
        for case in tqdm(self.TEST_IMAGES):
            samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
            for sample in tqdm(samples, desc=case):
                imagePath = os.path.join(self.IMAGE_DIR, case, os.path.splitext(sample)[0] + ".tif")
                image = cv2.imread(imagePath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = self.desc.describe(gray)
                label_path = os.path.join(self.GT_DIR, case, os.path.splitext(sample)[0] + ".csv")
                label = 1 if os.path.isfile(label_path) else 0
                prediction = self.model.predict(hist.reshape(1, -1))
                self.TEST_LABELS.append(label)
                self.PRED_LABELS.append(prediction)


    def predict(self, PRED_PATH , MODEL_PATH):
        model_clone = joblib.load(MODEL_PATH)
        image = cv2.imread(PRED_PATH)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = self.desc.describe(gray)
        prediction = model_clone.predict(hist.reshape(1, -1))
        return prediction


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
    args = vars(ap.parse_args())



    mitoses_model = recognize(args["data"],args["gt"])
    mitoses_model.prepare()
    mitoses_model.train()
    mitoses_model.test()
    mitoses_model.eval()








