# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:49:04 2021

@author: jai
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16


# Read input images and assign labels based on folder names
print(os.listdir("C:/Users/91944/Documents/JKsurya BIT/research paper/Covid Detection/Code/Dataset-created"))

SIZE = 256  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("C:/Users/91944/Documents/JKsurya BIT/research paper/Covid Detection/Code/Dataset-created/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.PNG")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels into respective lists

test_images = []
test_labels = [] 
for directory_path in glob.glob("C:/Users/91944/Documents/JKsurya BIT/research paper/Covid Detection/Code/Dataset-created/validation/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.PNG")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

###################################################################
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#############################
#Load model wothout classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False

VGG_model.trainable = True

set_trainable = False
for layer in VGG_model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0


#Now, let us use features from convolutional network for svm
feature_extractor=VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_svm = features

from sklearn import svm
svm_model = svm.SVC(kernel = 'linear', C = 1)


svm_model.fit(X_for_svm, y_train)

#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)


prediction_svm = svm_model.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_svm = le.inverse_transform(prediction_svm)

'''
#Naive Bayes

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_for_svm, y_train)

prediction_nb = classifier.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_nb = le.inverse_transform(prediction_nb)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_svm))

#GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6]}
grid = GridSearchCV(svm.SVC(),param_grid)
grid.fit(X_for_svm, y_train)

#Bayesian Optimization

svm_opt(train_data, train_label, test_data, test_label,
  gamma_range = c(10^(-3), 10^1), cost_range = c(10^(-2), 10^2),
  svm_kernel = "radial", degree_range = c(3L, 10L),
  coef0_range = c(10^(-1), 10^1), init_points = 4, n_iter = 10,
  acq = "ei", kappa = 2.576, eps = 0, optkernel = list(type =
  "exponential", power = 2))
'''

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_svm))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_svm)
print(cm)
sns.heatmap(cm, annot=True)
'''
# outcome values order in sklearn
tp, fn, fp, tn = confusion_matrix(test_labels,prediction_svm,labels=[1,0])
print('Outcome values : \n', tp, fn, fp, tn)
'''
# classification report for precision, recall f1-score and accuracy
matrix = classification_report(test_labels,prediction_svm,labels=[1,0])
print('Classification report : \n',matrix)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_svm = svm_model.predict(input_img_features)[0] 
prediction_svm = le.inverse_transform([prediction_svm])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_svm)
print("The actual label for this image is: ", test_labels[n])



