import PIL
from PIL import Image

from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model

import numpy as np
from numpy import asarray, savez_compressed
from pickle import load

import os
from os import listdir, mkdir
from os.path import isdir, join, exists
import itertools

from helpers import standardize
from Classifier.classifier import Classifier

class FaceReg:
    def __init__(self, model_path="keras/model/facenet_keras.h5", classifier_dir="keras/svm", thres=0):
        self.detector = MTCNN()
        self.model = load_model(model_path)
        self.classifier = None
        self.classifier_dir = classifier_dir
        self.thres = thres

    def predict(self, img):
        """
        Method to predict the identity of list of images
        Args:
            img: list of images

        Returns:

        """
        if not self.classifier:
            path = join(self.classifier_dir, 'model.p')
            print("Loading classifier")
            if exists(path):
                with open(path, 'rb') as f:
                    self.classifier = load(f)
            else:
                raise Exception("Classifier is not provided, cannot make predictions.")

        if not isinstance(img, list):
            img = [img]

        faces = list(map(self._extract_face, img))
        # Flatten the list
        faces = list(itertools.chain.from_iterable(faces))

        face_embeddings = self._get_embeddings(faces)
        pred = self.classifier.predict(face_embeddings)

        return pred

    def add_new_identity(self, img, name, directory, re_train=True, **kwargs):
        """
            A method to add a new identity to the existing database.
        Args:
            img: path to image
            name:
            directory:
            re_train:
            **kwargs:

        Returns:

        """

        if not isinstance(img, list):
            img = [img]

        new_person_dir = join(directory, name)
        if not isdir(new_person_dir)
            mkdir(join(directory, name))

        for idx, i in enumerate(img):
            if not isinstance(i, PIL.JpegImagePlugin.JpegImageFile)\
                    or not isinstance(img, PIL.PngImagePlugin.PngImageFile):
                img = Image.fromarray(img)
            path = join(new_person_dir, "{}.jpg".format(idx))
            img.save(path)
        if re_train:
            self.train_classifer(directory, **kwargs)

        return

    def train_classifer(self, directory, **kwargs):
        X, y = self._load_dataset(directory, **kwargs)
        embeddings = self._get_embeddings(X)
        self.classifier = Classifier(save_dir=self.classifier_dir, thres=self.thres)
        self.classifier.train(embeddings, y)

        return

    def open_images(self, path):
        """
            A method to open image file(s) in the format required by FaceReg
        Args:
            path: path to directory or file to open as PIL Image file.

        Returns: a list of image(s)

        """
        def image_file(file):
            return file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

        if isdir(path):
            return [Image.open(join(path, file)) for file in os.listdir(path) if image_file(file)]
        elif os.path.exists(path):
            try:
                return [Image.open(path)]
            except IOError as e:
                print(e)
        else:
            raise FileNotFoundError

    def _extract_face(self, img, require_size=(160, 160)):
        """
        Method to extract a list of faces from a given photo.
        Args:
            img: img array
            is_rgb: boolean, whether the image is in rgb format. Default to True
            require_size: The require_size of the resulting face images

        Returns: list of images

        """
        if not isinstance(img, PIL.JpegImagePlugin.JpegImageFile) \
                or not isinstance(img, PIL.PngImagePlugin.PngImageFile):
            try:
                img = Image.fromarray(img)
            except TypeError as e:
                print(type(img))

        img = img.convert('RGB')

        pixels = asarray(img)

        results = self.detector.detect_faces(pixels)

        def get_face(res):
            nonlocal pixels

            x1, y1, w, h = res['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + w, y1 + h

            # Extract face from image
            face = pixels[y1:y2, x1:x2]
            #resize pixels to model's required size
            resized = Image.fromarray(face)
            resized = resized.resize(require_size)

            return asarray(resized) # return the face as a numpy array

        faces = list(map(get_face, results))

        return faces

    def _get_embeddings(self, faces):
        """
        Method to get face embeddings from a list of face images
        Args:
            faces: list of face images

        Returns: list of embeddings

        """
        # if not isinstance(faces, np.ndarray):
        #     print("Goddammit")
        #     faces = [faces]

        faces = list(map(standardize, faces))
        faces = asarray(faces)

        return self.model.predict(faces)

    def _load_faces(self, directory, is_train=True, **kwargs):
        faces = []

        for filename in listdir(directory):
            path = join(directory, filename)
            face = Image.open(path)
            face = self._extract_face(face, **kwargs)

            if is_train and len(face) != 0:
                face = [face[0]]

            faces += face


        return faces

    def _load_dataset(self, directory, save_process=False, saved_file_name=None, verbose=1, **kwargs):
        X, y = [], []

        for subdir in listdir(directory):
            path = join(directory, subdir)

            if not isdir(path):
                continue

            faces = self._load_faces(path, **kwargs)
            labels = [subdir]*len(faces)

            if verbose:
                print('>loaded %d examples for class: %s' % (len(faces), subdir))

            X.extend(faces)
            y.extend(labels)

        if save_process:
            if not saved_file_name:
                raise ValueError("Must specify saved_file_name if save_process is True")

            else:
                savez_compressed(saved_file_name, X, y)
        return asarray(X), asarray(y)




def main():
    # This point to the directory contains all the training data
    data_dir = '5-celebrity-faces-dataset/train'
    # Initialize a new object
    facereg = FaceReg()
    # This is for training a classifier on custom dataset. Comment out this line if it is not needed.
    facereg.train_classifer(data_dir)
    # Initialize a list to hold eval images
    images = []
    # Path point to the eval folder
    eval_path = join(data_dir, '../val/ben_afflek')
    # Loads all images from folder into image list
    for filename in listdir(eval_path):
        images.append(Image.open(join(eval_path, filename)))
    # Predict on list of eval images
    res = facereg.predict(images)
    # Print predictions
    print(res)
    return

if __name__ == '__main__':
    main()

