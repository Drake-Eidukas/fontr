import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from random import shuffle
import pickle
from collections import OrderedDict
import operator
import json

class Preprocessor(object):

    def __init__(self, img_path):
        self._img = cv2.imread(img_path)
    
    def get_img(self):
        return self._img

    def get_contours(self):
        '''
        returns contours of img
        '''
        thresh_img = self.otsu_threshold()
        # don't change thresh_img, we need it below
        img_for_contour = thresh_img.copy()
        _,contours,_ = cv2.findContours(img_for_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sub_imgs = []
        img_height, img_width = thresh_img.shape
        for contour in contours:
            # get rectangle contour is in
            x, y, w, h = cv2.boundingRect(contour)
            # if contours is too small it's probably garbage
            #if w * h < 0.0001 * (img_height * img_width):
                #continue
            sub_img = thresh_img[y:y+h, x:x+w]
            sub_img = cv2.resize(sub_img, (128, 128), interpolation=cv2.INTER_CUBIC)
            sub_imgs.append([sub_img, x, y, w, h])
        return sub_imgs

    def otsu_threshold(self):
        '''
        Applies otsu's threshold to an image and returns the image
        '''
        gray_img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.fastNlMeansDenoising(gray_img, None, 10)
        ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY +
                                   cv2.THRESH_OTSU)
        return thresh_img



class SVM(object):

    def __init__(self, train_img_dir):
        #self.model = LinearSVC(C=0.03125, class_weight='balanced')
        with open('weights4.pkl', 'rb') as clf_pkl:
            self.model = pickle.load(clf_pkl)
        #self.x, self.y = self.get_x_and_y(train_img_dir)
    
    def get_x_and_y(self, train_img_dir):
        '''
        Given a directory of training images, return a list
        containing features for each image as well as a list
        containing the labels of each image
        '''
        train_img_names = os.listdir(train_img_dir)
        shuffle(train_img_names)
        x = []
        y = []
        num = 0
        for train_img_name in train_img_names:
            print(train_img_name, num)
            train_img = cv2.imread(train_img_dir + train_img_name)
            train_img = cv2.resize(train_img, (128, 128), interpolation=cv2.INTER_CUBIC)
            train_img = self.otsu_threshold(train_img)
            features = self.get_features(train_img)
            x.append(features)
            # if it's a character
            if len(train_img_name) > 9:
                y.append(1)
            else:
                y.append(0)
            num += 1
        return x, y

    def train(self):
        '''
        Train the linear support vector machine and put in pickle file
        '''
        self.model.fit(self.x, self.y)
        with open('weights4.pkl', 'wb') as clf_pkl:
            pickle.dump(self.model, clf_pkl)

    # not really needed
    def train_and_score(self, train_img_dir):
        train_img_names = os.listdir(train_img_dir)
        shuffle(train_img_names)
        x = []
        y = []
        for train_img_name in train_img_names[:10000]:
            train_img = cv2.imread(train_img_dir + train_img_name)
            train_img = cv2.resize(train_img, (125, 125), cv2.INTER_CUBIC)
            train_img = self.otsu_threshold(train_img)
            x.append(self.get_features(train_img))
            # if it's a character
            if len(train_img_name) > 8:
                y.append(1)
            else:
                y.append(0)
        self.model.fit(x, y)

        # testing
        labels = []
        features = []
        for img_name in train_img_names[10000:]:
            if len(img_name) > 8:
                labels.append(1)
            else:
                labels.append(0)
            img = cv2.imread(train_img_dir + img_name)
            img = cv2.resize(img, (125, 125), cv2.INTER_CUBIC)
            img = self.otsu_threshold(img)
            features.append(self.get_features(img))
        print(self.model.score(features, labels))

    def extract_characters(self, img_name):
        '''
        Given an image, extract sub images likely to contain characters,
        draw boxes around these sub images, and return a list of these images
        '''
        preprocessor = Preprocessor(img_name)
        # for drawing boxes around suspected characters
        box_img = preprocessor.get_img()
        sub_imgs = preprocessor.get_contours()
        # list of likely characters
        char_imgs = []
        for sub_img in sub_imgs:
            features = self.get_features(sub_img[0])
            prediction = self.model.decision_function([features])
            x, y = sub_img[1], sub_img[2]
            w, h = sub_img[3], sub_img[4]
            # if it's a character, slight bias towards non character
            if prediction > 0.25:
                char_img = cv2.resize(sub_img[0], (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(box_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                char_imgs.append(char_img)
            else:
                cv2.rectangle(box_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.imwrite('boxes.png', box_img)
        return char_imgs

    def get_features(self, img):
        '''
        Given an image, return its features
        (histogram of oriented gradients)
        '''
        hog = cv2.HOGDescriptor()
        # flatten column vector into row vector
        h = hog.compute(img)
        np_array = np.array(h)
        h = np_array.T
        return np.array(h)[0].tolist()

    def otsu_threshold(self, img):
        '''
        Given an image, apply otsu's threshold to the image
        and return the image
        '''
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_img = cv2.fastNlMeansDenoising(gray_img, None, 10)
        ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY +
                                   cv2.THRESH_OTSU)
        return gray_img

def classify_letters(images):

    # Load the classifier, class names, scaler, number of clusters and vocabulary 
    clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FastFeatureDetector_create(40)
    des_ext = cv2.xfeatures2d.FREAK_create()

    # List where all the descriptors are stored
    des_list = []


    for im in images:
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        if des != None:
            des_list.append(des)   

    des_list = np.asarray(des_list)
    # 
    test_features = np.zeros((len(des_list), k), "float32")
    for i in xrange(len(des_list)):
        words, distance = vq(des_list[i],voc)
        for w in words:
            test_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(des_list)+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scale the features
    test_features = stdSlr.transform(test_features)

    # Perform the predictions
    predictions =  [classes_names[i] for i in clf.predict(test_features)]
    
    return to_json(sum_predictions(predictions))

def sum_predictions(predictions):
    
    d = dict()
    total = len(predictions)
    for prediction in predictions:
        if prediction in d:
            d[prediction] += 1.0
        else:
            d[prediction] = 1.0
    maximum = 0    
    for key in d:
        d[key] /= total
        if d[key] > maximum:
            maximum = d[key]
    rat = .95/maximum
    for key in d:
        d[key] *= rat
    return d
    
def to_json(frame_dict):
    """
    Given a dictionary of {'label':'popularity'...}, return json as specified earlier.
    """
    sorted_dict = OrderedDict(sorted(frame_dict.items(), key=operator.itemgetter(1), reverse=True))
    dict_list = []
    for key in sorted_dict:
        key = key.replace('_',' ') 
        d = {}
        d['font']=key
        d['popularity']=sorted_dict[key.replace(' ','_')]
        #d[key] = sorted_dict[key]
        dict_list.append(d)
    return json.dumps({'data': dict_list})

def blackbox(image_name):
    svm = SVM(None)
    return classify_letters(svm.extract_characters(image_name))

