# import the necessary packages
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from extract_feature.lbp import LocalBinaryPatterns
from extract_feature.sift import SIFT
from extract_feature.hog import HOG
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, confusion_matrix, auc
from tqdm import tqdm
import argparse
import cv2
import os
import random
from sklearn.externals import joblib
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np


class recognize:
    def __init__(self, image_dir, gt_dir):
        self.IMAGE_DIR = image_dir
        self.GT_DIR = gt_dir
        self.TRAIN_IMAGES = ['09', '28', '22', '50', '61', '62', '63', '64', '65', '66', '67', '68', '69', '34', '04',
                             '02', '03', '26', '01', '06', '07', '48', '49', '46', '47', '08', '45', '42', '29', '40',
                             '41', '05', '23', '24', '56', '25', '15', '12', '72', '71', '70', '20', '21', '11', '13',
                             '38', '59', '58', '17', '16', '19', '18', '31', '30', '51', '36', '53', '52', '33', '55',
                             '37', '32', '44']
        self.TEST_IMAGES = ['57', '60', '27', '14', '54', '73', '10', '35', '43', '39']
        self.TRAIN_LABELS = []
        self.TEST_LABELS = []
        self.PRED_SCORES = []
        self.TEST_NUM = 10
        self.PRED_LABELS = []
        self.TIMESTR = time.strftime("%Y%m%d-%H%M%S")

        # FEATURE MODEL
        self.LBP_points = 8
        self.LBP_radius = 1
        self.num_codes = 256
        self.iter = 1
        self.desc = LocalBinaryPatterns(self.LBP_points, self.LBP_radius)
        self.hog = HOG((8, 8), (2, 2), 9)
        self.sift = SIFT(self.num_codes, self.iter)

        # SVM model
        self.model = LinearSVC(C=100.0, random_state=42)
        self.classifier = svm.SVC(kernel='linear', verbose=False)
        self.scores = None
        self.KFOLD = 10

    def prepare(self):
        print("Data DIR:", os.path.join(self.IMAGE_DIR))
        print("GroundTruth DIR:", os.path.join(self.GT_DIR))
        images_list = next(os.walk(os.path.join(self.IMAGE_DIR)))[1]
        print("Image List: ", len(images_list))

    def train_LBP(self):
        print("******* Training ********")
        train_labels = []
        lbn_feat = []
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
                lbn_feat.append(hist)

        # train a Linear SVM on the data
        scoring = ['precision_macro', 'recall_macro', 'f1_macro']
        self.scores = cross_validate(self.model, lbn_feat, train_labels, scoring=scoring,
                                     cv=self.KFOLD,
                                     return_train_score=True, n_jobs=16, verbose=1, return_estimator=True)
        # print (sorted(self.scores.keys()))
        # print (self.scores['estimator'], self.scores['train_f1_macro'], self.scores['test_f1_macro'])
        f1 = self.scores['test_f1_macro']
        # print (type(f1), f1)
        idx = np.argmax(f1)
        models = self.scores['estimator']
        self.model = models[idx]
        # export trained model
        joblib.dump(self.model, 'logs/lbp_svm_' + self.TIMESTR + '.pkl', compress=9)

    def test_LBP(self):
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
                y_score = self.model.decision_function(hist.reshape(1, -1))
                self.TEST_LABELS.append(label)
                self.PRED_LABELS.append(prediction)
                self.PRED_SCORES.append(y_score[0])

    def train_HOG(self):
        print("******* Training ********")
        train_labels = []
        hog_feat = []
        for case in tqdm(self.TRAIN_IMAGES):
            samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
            for sample in tqdm(samples, desc=case):
                imagePath = os.path.join(self.IMAGE_DIR, case, os.path.splitext(sample)[0] + ".tif")
                image = cv2.imread(imagePath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = self.hog.hog(gray)
                label_path = os.path.join(self.GT_DIR, case, os.path.splitext(sample)[0] + ".csv")
                lable = 1 if os.path.isfile(label_path) else 0
                train_labels.append(lable)
                hog_feat.append(hist)

        # train a Linear SVM on the data
        scoring = ['precision_macro', 'recall_macro', 'f1_macro']
        self.scores = cross_validate(self.model, hog_feat, train_labels, scoring=scoring,
                                     cv=self.KFOLD,
                                     return_train_score=True, n_jobs=16, verbose=1, return_estimator=True)
        # print (sorted(self.scores.keys()))
        # print (self.scores['estimator'], self.scores['train_f1_macro'], self.scores['test_f1_macro'])
        f1 = self.scores['test_f1_macro']
        # print (type(f1), f1)
        idx = np.argmax(f1)
        models = self.scores['estimator']
        self.model = models[idx]
        # export trained model
        joblib.dump(self.model, 'logs/hog_svm_' + self.TIMESTR + '.pkl', compress=9)

    def test_HOG(self):
        print("******* Testing ********")
        # loop over the testing images
        for case in tqdm(self.TEST_IMAGES):
            samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
            for sample in tqdm(samples, desc=case):
                imagePath = os.path.join(self.IMAGE_DIR, case, os.path.splitext(sample)[0] + ".tif")
                image = cv2.imread(imagePath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = self.hog.hog(gray)
                label_path = os.path.join(self.GT_DIR, case, os.path.splitext(sample)[0] + ".csv")
                label = 1 if os.path.isfile(label_path) else 0
                prediction = self.model.predict(hist.reshape(1, -1))
                y_score = self.model.decision_function(hist.reshape(1, -1))
                self.TEST_LABELS.append(label)
                self.PRED_LABELS.append(prediction)
                self.PRED_SCORES.append(y_score[0])



    def train_sift(self):
        print("******* Training using SIFT Features ********")
        train_labels = []
        paths = []
        for case in (self.TRAIN_IMAGES):
            samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
            for sample in (samples):
                imagePath = os.path.join(self.IMAGE_DIR, case, os.path.splitext(sample)[0] + ".tif")
                # image_paths.append((imagePath)
                # image = cv2.imread(imagePath)
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # hist = self.desc.describe(gray)
                label_path = os.path.join(self.GT_DIR, case, os.path.splitext(sample)[0] + ".csv")
                paths.append((imagePath, label_path))
            # label = 1 if os.path.isfile(label_path) else 0
            # train_labels.append(label)

        sift_feat, self.TRAIN_LABELS = self.sift.extract(paths)
        unique, counts = np.unique(self.TRAIN_LABELS, return_counts=True)
        print ('unique, counts', unique, counts)
        # train a Linear SVM on the data
        scoring = ['precision_macro', 'recall_macro', 'f1_macro']
        self.scores = cross_validate(self.classifier, sift_feat, self.TRAIN_LABELS, scoring=scoring, cv=self.KFOLD,
                                     return_train_score=True, n_jobs=16, verbose=1, return_estimator=True)
        # print (sorted(self.scores.keys()))
        # print (self.scores['estimator'], self.scores['train_f1_macro'], self.scores['test_f1_macro'])
        f1 = self.scores['test_f1_macro']
        # print (type(f1), f1)
        idx = np.argmax(f1)
        models = self.scores['estimator']
        self.classifier = models[idx]
        # print (scores[
        # self.classifier.fit(sift_feat, train_labels)
        # export trained model
        joblib.dump(self.classifier, 'logs/sift_svm_' + self.TIMESTR + '.pkl', compress=9)



    def test_sift(self):
        print("******* Testing ********")
        # loop over the testing images
        for case in tqdm(self.TEST_IMAGES):
            samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
            for sample in tqdm(samples, desc=case):
                imagePath = os.path.join(self.IMAGE_DIR, case, os.path.splitext(sample)[0] + ".tif")
                # image = cv2.imread(imagePath)
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # hist = self.desc.describe(gray)
                label_path = os.path.join(self.GT_DIR, case, os.path.splitext(sample)[0] + ".csv")
                # label = 1 if os.path.isfile(label_path) else 0
                # print("******* generating sift feature ********")
                des, train_labels = self.sift.get_features(imagePath, label_path)
                # print("******* generating sift histogram ********")
                sift_feat = self.sift.gen_histogram(des)
                # print("******* predicting ********")
                prediction = self.classifier.predict(sift_feat.reshape(1, -1))
                y_score = self.classifier.decision_function(sift_feat.reshape(1, -1))
                self.TEST_LABELS.append(train_labels)
                self.PRED_LABELS.append(prediction[0])
                self.PRED_SCORES.append(y_score[0])

    def predict(self, PRED_PATH, MODEL_PATH):
        model_clone = joblib.load(MODEL_PATH)
        image = cv2.imread(PRED_PATH)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = self.desc.describe(gray)
        prediction = model_clone.predict(hist.reshape(1, -1))
        return prediction

    def eval(self):
        (precision, recall, fscore, support) = score(self.TEST_LABELS, self.PRED_LABELS)
        conf_mat = confusion_matrix(self.TEST_LABELS, self.PRED_LABELS)
        (tn, fp, fn, tp) = conf_mat.ravel()
        # print (self.TEST_LABELS)
        # print (self.PRED_LABELS)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        print('(tn, fp, fn, tp): {}'.format((tn, fp, fn, tp)))

        tr_unique, tr_counts = np.unique(self.TRAIN_LABELS, return_counts=True)
        tst_unique, tst_counts = np.unique(self.TEST_LABELS, return_counts=True)

        myCsvRow = '\n' + str(self.num_codes) + ',' + \
                   ','.join(str(x) for x in tr_counts) + \
                   ',' + ','.join(str(x) for x in tst_counts) + ',' + \
                   str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(tp) + ',' + \
                   str(precision) + ',' + str(recall) + ',' + str(fscore) + ',' + str(support)

        with open('logs/METRICS.csv', 'a') as fd:
            fd.write(myCsvRow)

    def plotROC(self):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # print (self.PRED_SCORES) , pos_label=
        fpr, tpr, _ = roc_curve(self.TEST_LABELS, self.PRED_SCORES)
        roc_auc = auc(fpr, tpr)
        print ('ROC AUC: {}'.format(roc_auc))

        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(self.TEST_LABELS.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # # Plot of a ROC curve for a specific class

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('logs/roc_curve_' + self.TIMESTR + '.png')
        plt.close()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="path to the training images")
    ap.add_argument("-gt", "--gt", required=True,
                    help="path to the tesitng images")
    ap.add_argument("-sift", "--sift", required=False,
                    help="path to the tesitng images")
    args = vars(ap.parse_args())

    mitoses_model = recognize(args["data"], args["gt"])
    # mitoses_model = recognize('../../DATA/mitosis_image_data/', '../../DATA/mitoses_ground_truth/')

    mitoses_model.prepare()

    mitoses_model.train_HOG()
    mitoses_model.test_HOG()
    mitoses_model.eval()
    mitoses_model.plotROC()
