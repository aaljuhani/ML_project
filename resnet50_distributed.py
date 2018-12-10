# import the necessary packages
import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(42)
from extract_feature.lbp import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import cv2
import os
import random
import time
import pickle
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from tensorflow.python.keras import backend as K
import functools
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import SGD
import horovod.keras as hvd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class recognize:
    def __init__(self, image_dir, gt_dir, lbn_points, lbn_radius):
        self.IMAGE_DIR = image_dir
        self.GT_DIR = gt_dir
        self.TRAIN_IMAGES = []
        self.TEST_IMAGES = []
        self.TRAIN_LABELS = []
        self.TEST_LABELS = []
        self.TEST_NUM = 10
        self.PRED_LABELS = []
        self.lbn_data = None
        self.image_path_list = None
        self.image_arr = None
        self.train_image_resize = None
        self.test_image_resize = None

        self.test_lbn_data = None
        self.test_image_path_list = None
        self.test_image_arr = None

        # FEATURE MODEL
        self.desc = LocalBinaryPatterns(int(lbn_points), int(lbn_radius))
        # SVM model
        self.model = LinearSVC(C=100.0, random_state=42)
        self.nn_model = None

    def prepare(self):
        print("Data DIR:", os.path.join(self.IMAGE_DIR))
        print("GroundTruth DIR:", os.path.join(self.GT_DIR))
        images_list = next(os.walk(os.path.join(self.IMAGE_DIR)))[1]
        print("Image List: ", len(images_list))
        self.TEST_IMAGES = random.sample(images_list, self.TEST_NUM)
        self.TRAIN_IMAGES = list(set(images_list) - set(self.TEST_IMAGES))
        self.TRAIN_IMAGES = np.asarray(self.TRAIN_IMAGES)
        self.TEST_IMAGES = np.asarray(list(set(self.TEST_IMAGES)))

    def train(self, model):
        print("******* Processing Training data********")
        train_labels = []
        lbn_data = []
        image_path_list = []
        image_arr = []

        if os.path.exists("./pickle/lbn_data.p"):
            self.lbn_data = pickle.load(open("./pickle/lbn_data.p", "rb"))
            self.TRAIN_LABELS = pickle.load(open("./pickle/train_labels.p", "rb"))
            self.image_path_list = pickle.load(open("./pickle/image_path_list.p", "rb"))
            self.image_arr = pickle.load(open("./pickle/image_arr.p", "rb"))
        else:

            for case in tqdm(self.TRAIN_IMAGES):
                samples = next(os.walk(os.path.join(self.IMAGE_DIR, case)))[2]
                for sample in tqdm(samples, desc=case):
                    imagePath = os.path.join(self.IMAGE_DIR, case, os.path.splitext(sample)[0] + ".tif")
                    image = cv2.imread(imagePath)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    hist = self.desc.describe(gray)
                    label_path = os.path.join(self.GT_DIR, case, os.path.splitext(sample)[0] + ".csv")
                    label = 1 if os.path.isfile(label_path) else 0
                    train_labels.append(label)
                    lbn_data.append(hist)
                    image_path_list.append(imagePath)
                    #image_arr.append(gray)
                    image_arr.append(image)

            lbn_data = np.asarray(lbn_data)
            train_labels = np.asarray(train_labels)
            image_path_list = np.asarray(image_path_list)
            image_arr = np.asarray(image_arr)

            self.TRAIN_LABELS = np.asarray(train_labels)
            self.lbn_data = lbn_data
            self.image_path_list = image_path_list
            self.image_arr = image_arr

            pickle.dump(lbn_data, open("./pickle/lbn_data.p", "wb"))
            pickle.dump(train_labels, open("./pickle/train_labels.p", "wb"))
            pickle.dump(image_path_list, open("./pickle/image_path_list.p", "wb"))
            pickle.dump(image_arr, open("./pickle/image_arr.p", "wb"))

        if model == 'svm':
            #train a Linear SVM on the data
            self.model.fit(lbn_data, train_labels)
        elif model == 'nn':
            self.nn_train()

    def test(self, model):
        print("******* Processing Validation data ********")
        test_labels = []
        lbn_data = []
        image_path_list = []
        image_arr = []

        if os.path.exists("./pickle/test_lbn_data.p"):
            self.test_lbn_data = pickle.load(open("./pickle/test_lbn_data.p", "rb"))
            self.TEST_LABELS = pickle.load(open("./pickle/test_labels.p", "rb"))
            self.test_image_path_list = pickle.load(open("./pickle/test_image_path_list.p", "rb"))
            self.test_image_arr = pickle.load(open("./pickle/test_image_arr.p", "rb"))
        else:
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
                    test_labels.append(label)
                    lbn_data.append(hist)
                    image_path_list.append(imagePath)
                    #image_arr.append(gray)
                    image_arr.append(image)
                    if(model == 'svm'):
                        prediction = self.model.predict(hist.reshape(1, -1))
                        self.TEST_LABELS.append(label)
                        self.PRED_LABELS.append(prediction)

            lbn_data = np.asarray(lbn_data)
            test_labels = np.asarray(test_labels)
            image_path_list = np.asarray(image_path_list)
            image_arr = np.asarray(image_arr)
            print("real!: {}".format(image_arr[0].shape))

            self.TEST_LABELS = np.asarray(test_labels)
            self.test_lbn_data = lbn_data
            self.test_image_path_list = image_path_list
            self.test_image_arr = image_arr

            pickle.dump(lbn_data, open("./pickle/test_lbn_data.p", "wb"))
            pickle.dump(test_labels, open("./pickle/test_labels.p", "wb"))
            pickle.dump(image_path_list, open("./pickle/test_image_path_list.p", "wb"))
            pickle.dump(image_arr, open("./pickle/test_image_arr.p", "wb"))


    def eval(self):
        precision, recall, fscore, support = score(self.TEST_LABELS, self.PRED_LABELS)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

    def generator(self, x, y, batch_size):
       batch_x = np.zeros((batch_size,224,224,3))
       batch_y = np.zeros((batch_size,1))

       while True:
         for i in range(batch_size):
            idx = random.randrange(len(x))
            batch_x[i] = x[idx]
            batch_y[i] = y[idx]
         yield (batch_x, batch_y)

    def nn_train(self):
        # Horovod: initialize Horovod.
        hvd.init()

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))


        self.test('nn')
        train_image = mitoses_model.image_arr
        #print("train_image {}".format(train_image))
        print("train_img_shape {}".format(train_image[0].shape))
        if os.path.exists("./pickle/train_image_resize_224.p"):
            train_image_resize = pickle.load(open("./pickle/train_image_resize_224.p", "rb"))
            self.train_image_resize = train_image_resize
        else:
            train_image_resize = np.zeros((train_image.shape[0], 224, 224, 3))
            for i in range(train_image.shape[0]):
                each_train_image = np.array([cv2.resize(train_image[i], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)])
                train_image_resize[i] = each_train_image
            pickle.dump(train_image_resize, open("./pickle/train_image_resize_224.p", "wb"))
        print("train_image_value_reshape: {} ".format(train_image_resize[0].shape))
        #print("train_image_value: {} ".format(train_image_resize))
        print("num_occurrences_each_label_train: {}".format(np.bincount(self.TRAIN_LABELS)))
        #print("img_shape: {}".format(train_image[0].shape))
        #print("img_shape_ind_0: {}".format(train_image[0].shape[0]))
        #train_image = np.reshape(train_image, (562, train_image[, -1, 3))

        test_image = mitoses_model.test_image_arr
        #print("test_image {}".format(test_image))
        print("test_img_shape {}".format(test_image[0].shape))

        if os.path.exists("./pickle/test_image_resize_224.p"):
            test_image_resize = pickle.load(open("./pickle/test_image_resize_224.p", "rb"))
            self.test_image_resize = test_image_resize
        else:
            test_image_resize = np.zeros((test_image.shape[0], 224, 224, 3))
            for i in range(test_image.shape[0]):
                each_test_image = np.array([cv2.resize(test_image[i], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)])
                print("each_test_img: {}".format(each_test_image.shape))
                test_image_resize[i] = each_test_image
            pickle.dump(test_image_resize, open("./pickle/test_image_resize_224.p", "wb"))

        print("test_image_value_reshape: {} ".format(test_image_resize[0].shape))
        #print("test_image_value: {} ".format(test_image_resize))
        print("num_occurrences_each_label_test: {}".format(np.bincount(self.TEST_LABELS)))

        train_image = train_image_resize 
        train_label = self.TRAIN_LABELS
        #print("train_label: {}".format(train_label))

        test_image = test_image_resize
        test_label = self.TEST_LABELS

        #print("num_images: {}".format(train_image.shape))
        #print("shape: {}".format(train_image[0].shape))
        # Define a model
        input = Input(shape=(224, 224, 3))
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        base_model.summary()

        output = base_model(input)
        #print("output!!!!: {}".format(output))
        output = Flatten(name='flatten')(output)
        #print("flat!!!: {}".format(output))
        #output = GlobalAveragePooling2D()(output)
        output = Dense(1024, activation='relu')(output)
        #print("output_relu: {} ".format(output))
        pred = Dense(2, activation='softmax')(output)
        #print("output_softmax: {} ".format(pred))
        model = Model(inputs=input, outputs=pred)

        model.summary()

        # Disrtibued parameter update
        opt = SGD(lr=0.01 * hvd.size())
        opt = hvd.DistributedOptimizer(opt)
        # Configure optimizer, loss, and metrics for model
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        path_checkpoint = 'resnet50.keras'
        if hvd.rank() == 0:
           callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      verbose=1,
                                      save_weights_only=True)
           callback_tensorboard = TensorBoard(log_dir='./resnet50_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
        
           early_stopping = EarlyStopping(patience=10)
           metric = Metrics()
           callbacks = [callback_checkpoint, callback_tensorboard, early_stopping]
        else:
           callbacks = None
        try:
           model.load_weights(path_checkpoint)
        except Exception as error:
           print("Error trying to load checkpoint.")
           print(error)
       
        batch_size = 32

        # Fit the model
        start = time.time()
        result = model.fit_generator(self.generator(train_image,train_label,batch_size), validation_data=self.generator(test_image,test_label,batch_size), steps_per_epoch= (len(train_image)//batch_size)//hvd.size(), validation_steps = (3 * (len(test_image)//batch_size))//hvd.size(), epochs=30)
        self.nn_model = model

        end = time.time()
        print("Time to train: {}".format(end - start))
        # Plot losses
        fig, loss_ax = plt.subplots()
        loss_ax.set_title('Train_Validation_Loss')
        loss_ax.plot(result.history['loss'], 'y', label='train loss')
        loss_ax.plot(result.history['val_loss'], 'r', label='val loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper right')
        plt.savefig('loss.png')
        
        # Confusion matrix for training image
        pred_prob = model.predict(train_image, batch_size=1)
        pred_label = pred_prob > 0.5
        #print("pred_for_each_label: {}".format(pred_label))
        pred_label = np.argmax(pred_label, axis=1)
        #print("pred_label: {}".format(pred_label))
        true_label = train_label
        print("Training Confusion matrix: {}".format(confusion_matrix(true_label, pred_label)))

        # Conifusion matrix for test image
        pred_prob = model.predict(test_image, batch_size=1)
        pred_label = pred_prob > 0.5
        # print("pred_for_each_label: {}".format(pred_label))
        pred_label = np.argmax(pred_label, axis=1)
        # print("pred_label: {}".format(pred_label))
        true_label = test_label
        conf_mat = confusion_matrix(true_label, pred_label)
        print("Confusion matrix for test set: {}".format(conf_mat))

        # true positive rate(recall) for test image
        tpr = conf_mat[1][1] / np.sum(conf_mat[1])
        print("TPR: {}".format(tpr))
        # false positive rate for test image
        fpr = conf_mat[0][1] / np.sum(conf_mat[0])
        print("FPR: {}".format(fpr))
        # precision for test image
        pre = conf_mat[1][1] / np.sum([conf_mat[0][1], conf_mat[1][1]])
        print("precision: {}".format(pre))

        # F1 score for test image
        pred_prob = model.predict(test_image, batch_size=1)
        pred_label = pred_prob > 0.5
        pred_label = np.argmax(pred_label, axis=1)
        print("F-1 score for test set: {}".format(f1_score(test_label, pred_label)))

        # ROC_curve & AUC for test image
        pred_prob = model.predict(test_image, batch_size=1)
        #print("pred: {}".format(pred_prob)) 
        prob_pos = pred_prob[:,1]
        fpr, tpr, threshold = roc_curve(test_label, prob_pos)
        roc_auc = auc(fpr, tpr)
        print("threshold: {}".format(threshold))
        print("fpr_thres: {}".format(fpr))
        print("tpr_thres: {}".format(tpr))
        print("AUC: {}".format(roc_auc))
        fig2, auc_ax = plt.subplots()
        auc_ax.set_title('ROC Curve')
        auc_ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        auc_ax.legend(loc = 'lower right')
        auc_ax.plot([0, 1], [0, 1],'r--')
        auc_ax.set_ylabel('True Positive Rate')
        auc_ax.set_xlabel('False Positive Rate')
        plt.savefig('ROC_curve.png')

        # save weights of models
        #model.save_weights("resnet50.keras")

    def as_keras_metric(self, method):
        @functools.wraps(method)
        def wrapper(self, args, **kwargs):
            """ Wrapper for turning tensorflow metrics into keras metrics """
            value, update_op = method(self, args, **kwargs)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([update_op]):
                value = tf.identity(value)
            return value
        return wrapper

class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        pred_label = self.model.predict(self.x, batch_size=32)
        pred_label = pred_label > 0.5
        pred_label = np.argmax(pred_label, axis=1)
        true_label = self.y
        self.precision_score=(true_label, pred_label)
        print("precision_score: {}".format(self.precision_score))
        return


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
    ap.add_argument("-model", "--model", required=True, 
                    help="svm or nn")
    args = vars(ap.parse_args())

    mitoses_model = recognize(args["data"], args["gt"], args["lbn_points"], args["lbn_radius"])
    mitoses_model.prepare()
    # if(args["model"]=="svm"):
    mitoses_model.train(args["model"])

    """
    print("num_train_images: {}".format(len(mitoses_model.TRAIN_IMAGES)))
    print("num_train_instances: {}".format(len(mitoses_model.lbn_data)))
    print("instance_shape: {}".format(mitoses_model.lbn_data[0].shape))
    print("instance[0]: {}".format(mitoses_model.lbn_data[0]))
    print("num_occurrences_each_label: {}".format(np.bincount(mitoses_model.TRAIN_LABELS)))
    print("first_10_labels: {}".format(mitoses_model.TRAIN_LABELS[:10]))
    print("images_path: {}".format(mitoses_model.image_path_list[:10]))
    print("image.shape: {}".format(mitoses_model.image_arr[0].shape))
    """


