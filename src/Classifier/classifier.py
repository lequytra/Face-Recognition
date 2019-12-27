import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder

from pickle import dump, load
from numpy import abs, amax, argmax

class Classifier:
    def __init__(self, save_dir="../keras/svm", thres=0):

        self.clf = SVC(kernel='linear', probability=True, verbose=1)

        self.save_dir = save_dir
        self.thres = thres
        self.out_encoder = None

    def train(self, trainX, trainy, testX=None, testy=None, verbose=True):

        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)

        if testX:
            testX = in_encoder.transform(testy)

        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(trainy)
        trainy = self.out_encoder.transform(trainy)
        if testy:
            testy = self.out_encoder.transform(testy)

        self.clf.fit(trainX, trainy)

        if verbose:
            train_pred = self.clf.predict(trainX)
            train_score = accuracy_score(train_pred, trainy)

            print("Train set Accuracy = %d" %train_score*100)

            if testX:
                test_pred = self.clf.predict(testX)
                test_score = accuracy_score(testy, test_pred)
                print("Test set Accuracy = %d" %test_score*100)
        model_path = os.path.join(self.save_dir, 'model.p')
        out_encoder_path = os.path.join(self.save_dir, 'label_index.p')
        with open(model_path, 'wb') as f:
            dump(self, f)

        print("Training finished!! Model file and label encoder are saved at " +
              "%s and %s" %(model_path, out_encoder_path))

        return

    def predict(self, X):
        if self.out_encoder is None:
            raise Exception("Classifier might not have been trained. Please checked again!!")
        # Normalize input
        in_encoder = Normalizer(norm='l2')
        X = in_encoder.transform(X)

        pred_prob = self.clf.predict_proba(X)
        print(pred_prob)
        pred_y = argmax(pred_prob, axis=1)

        pred_prob = amax(pred_prob, axis=1)

        pred_labels = self.out_encoder.inverse_transform(pred_y)

        pred_labels[pred_prob < self.thres] = "Unknown"
        pred_prob[pred_prob < self.thres] = abs(pred_prob[pred_prob < self.thres] - 1)

        return list(zip(pred_labels, pred_prob))