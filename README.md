# Face Recognition

### FaceReg:
#### Initializing the object:
The Facereg object takes in the following optional argument: 
- ```model_path```: a path to the a Keras face embedding model (e.g., facenet)
- ```classifier_path```: path to face classifier (e.g., SVM)
-  ```thres```: a threshold for predicting unknown. This is only used when a classifier is re-trained. 
This tells the classifier if the confidence score is below threshold, then classifer should return "unknown". 
```thres``` is default to 0. 

#### Public Methods: 
- ```train_classifier(directory, **kwargs)```:  The method takes in a path to the face database. It will load the images
in database, extract faces, get face embeddings and train the classifier on the identities available in the directory. 
The directory should be of the following structure:
    - Directory:
        - identity1:
            - image 1
            - image 2
            - image 3
        - identity2:
            - image 1
            - image 2
            
- ```open_images(path)```: path is a path string to an image file or a directory of image files. This will open them as 
the required image object(s) for ```FaceReg```. Return as list of image(s). 

- ```add_new_identity(img, name, directory, retrain=True, **kwargs)```: 
Method will create a new folder in the database folder and add the image to the new identity folder. Retrain and save a 
new classifier if needed. The classifier will be saved to ```classifier_dir```. 
    - ```img```: a PIL Image file (open file through ```open_images```)
    - ```name```: a string to identity the new identity with. 
    - ```directory```: path to database folder. 
    - ```re_train```: (default to True) whether to retrain and save a new classifier. 

- ```predict(img)```: Does what it is supposed to. Given an image or list of images, return a list
of tuples (predicted identity, conf-score) for the images. Again, open image via ```open_images```. 


**Note:**
- For best training result, it is best the the training images do not contain any other identity rather than the one 
indicated by its folder's name. 
- Only jpeg and png are supported at point of writing. 
- ```FaceReg``` is designed as a "plug and play", meaning one can try mutiple face embedding models that might or might 
not be better than the ```FaceNet``` provided. The catch is that model must be a Keras .h5 saved model and can be called
predict on. 
