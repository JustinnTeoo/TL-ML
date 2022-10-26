#%% Just for information

# Modules (Transfer Learning)
# densenet 
# imagenet_utils 
# inception_resnet_v2 
# inception_v3 
# mobilenet 
# mobilenet_v2 
# mobilenet_v3 
# nasnet 
# resnet 
# resnet50 
# resnet_v2 
# vgg16 
# vgg19 
# xception 

# Functions (Transfer Learning)
# DenseNet121
# DenseNet169
# DenseNet201
# InceptionResNetV2
# InceptionV3
# MobileNet
# MobileNetV2
# MobileNetV3Large
# MobileNetV3Small
# NASNetLarge
# NASNetMobile
# ResNet101
# ResNet101V2
# ResNet152
# ResNet152V2
# ResNet50
# ResNet50V2
# VGG16
# VGG19
# Xception

#%% Import Compulsory Libs

#imports
from pathlib import Path
#import numpy as np
#import joblib
#import theano
import tensorflow
import keras
#from tensorflow.keras import backend

from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
    
import numpy as np
import joblib
from PIL import Image

#%% Import Transfer Learning

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB6
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import preprocess_input

from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input

from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input

from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input

#%% Call Variables to use

# Refer to above cells on which variables to use
Modules = "inception_v3"
Functions = "InceptionV3"

#func2 selection
Modfunc = {
    'VGG16': VGG16,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201,
    'InceptionResNetV2': InceptionResNetV2,
    'InceptionV3': InceptionV3,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'NASNetLarge': NASNetLarge,
    'NASNetMobile': NASNetMobile,
    'ResNet101': ResNet101,
    'ResNet101V2': ResNet101V2,
    'ResNet152': ResNet152,
    'ResNet152V2': ResNet152V2,
    'ResNet50': ResNet50,
    'ResNet50V2': ResNet50V2,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'Xception':Xception
}

if Functions in Modfunc:
    func2 = Modfunc[Functions]
    func2()
    
#%% Input Image Size

size1 = ["DenseNet121","DenseNet169","DenseNet201","MobileNet",
"MobileNetV2","MobileNetV3Large","MobileNetV3Small","NASNetMobile","ResNet101",
"ResNet101V2","ResNet152","ResNet152V2","ResNet50","ResNet50V2","VGG16","VGG19","Xception"]
check1 = any(item in Functions for item in size1)
if check1 is True:
    new_width  = 224
    new_height = 224

size2 = ["InceptionResNetV2","InceptionV3"]
check2 = any(item in Functions for item in size2)
if check2 is True:
    new_width  = 299
    new_height = 299

size3 = ["NASNetLarge"]
check3 = any(item in Functions for item in size3)
if check3 is True:
    new_width  = 331
    new_height = 331
    
#%% Output Layer Shape 

mod1 = ["VGG16","VGG19"]
Mod1 = any(item in Functions for item in mod1)
if Mod1 is True:
    s1 = 7
    s2 = 7
    s3 = 512

mod2 = ["Xception","ResNet101","ResNet101V2","ResNet152","ResNet152V2",
"ResNet50","ResNet50V2"]
Mod2 = any(item in Functions for item in mod2)
if Mod2 is True:
    s1 = 7
    s2 = 7
    s3 = 2048

mod3 = ["InceptionV3"]
Mod3 = any(item in Functions for item in mod3)
if Mod3 is True:
    s1 = 8
    s2 = 8
    s3 = 2048

mod4 = ["InceptionResNetV2"]
Mod4 = any(item in Functions for item in mod4)
if Mod4 is True:
    s1 = 8
    s2 = 8
    s3 = 1536

mod5 = ["MobileNet","DenseNet121"]
Mod5 = any(item in Functions for item in mod5)
if Mod5 is True:
    s1 = 7
    s2 = 7
    s3 = 1024

mod6 = ["MobileNetV2","EfficientNetB0"]
Mod6 = any(item in Functions for item in mod6)
if Mod6 is True:
    s1 = 7
    s2 = 7
    s3 = 1280

mod7 = ["DenseNet169"]
Mod7 = any(item in Functions for item in mod7)
if Mod7 is True:
    s1 = 7
    s2 = 7
    s3 = 1664

mod8 = ["DenseNet201"]
Mod8 = any(item in Functions for item in mod8)
if Mod8 is True:
    s1 = 7
    s2 = 7
    s3 = 1920

mod9 = ["NASNetMobile"]
Mod9 = any(item in Functions for item in mod9)
if Mod9 is True:
    s1 = 7
    s2 = 7
    s3 = 1056

mod10 = ["NASNetLarge"]
Mod10 = any(item in Functions for item in mod10)
if Mod10 is True:
    s1 = 11
    s2 = 11
    s3 = 4032

#%% Input Data Image
#must be same folder with coding
RED_path = Path("RUN2")/"RED"
BLACK_path = Path("RUN2")/"BLACK"

images = []
labels = []

#Load all the RED images
for img in RED_path.glob("*.jpg"):
    # load image from disk
    img = image.load_img(img)
    
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    
    #convert into numpy array
    image_array = image.img_to_array(img)
    
    #add the image to the list of images
    images.append(image_array)
    
    #for each hemorrhage image, the expected value shoud be 1
    labels.append(0)

#Load all the BLACK images
for img in BLACK_path.glob("*.jpg"):
    # load image from disk
    img = image.load_img(img)
    
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    #convert int to numpy array
    image_array = image.img_to_array(img)
    
    #add the image to the list of images
    images.append(image_array)
    
    #for each no_hemorhhage image, the expected value shoud be 0
    labels.append(1)
    

#%% Create Numpy array

x_val = np.array(images)

#convert all the labels to a numpy array
y_val = np.array(labels)
num_classes = 2
from keras.utils import np_utils
y_val = np_utils.to_categorical(y_val, num_classes)

#normalize data 0 to 1 range
x_val = preprocess_input(x_val)

#load pretrained NN as feature extractor ... 3 - rgb
pretrained_nn = func2(weights="imagenet", include_top=False, input_shape=(new_width,new_height,3))

#%% Extract TL Features

features_x_val = pretrained_nn.predict(x_val)
features_flatten = features_x_val.reshape((features_x_val.shape[0],s1*s2*s3))
print(features_x_val.shape,features_flatten.shape)

#save the array of extracted features into a file
joblib.dump(features_x_val, f"x_val_new_{Functions}.dat")

#save the matching array of expected values to a file
joblib.dump(y_val, f"y_val_new_{Functions}.dat")

#%% Import sklearn

import joblib
#import pandas as pd
from sklearn.model_selection import GridSearchCV
#from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC # ML Model

features = features_flatten
lables = labels

from sklearn.model_selection import train_test_split
#First Split, splitting the datat to 60:40 ratio.
X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.3,random_state=42, stratify=labels)
#Second split, splitting the test data into half, i.e., validation and test.

#Dr, the stratification should follow y_test and not labels 

X_val,X_test,y_val,y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42,stratify=y_test)

#%% HyperParameter Fine Tuning
# input_shape = (64,64,3)

# def print_results(results):
#     print('Best Param: {}\n'.format(results.best_params_))
#     means = results.cv_results_['mean_test_score']
#     stds = results.cv_results_['std_test_score']
#     for mean, std, params in zip(means,stds, results.cv_results_['params']):
#         print('{} (+/-{}) for {}'.format(round(mean,3),round(std*2,3),params))

# svc=SVC()
# parameters={
#           'kernel':['linear','poly', 'rbf'],
#           'C':[0.01, 0.1, 1, 10, 100],
#           'gamma':[0.01, 0.1, 1, 10, 100]
# #        'kernel':['linear'],
# #        'C':[0.01],
# #        'gamma':[0.01]
#         }

# cv = GridSearchCV(svc,parameters,cv=2)

# import joblib
# cv.fit(X_train,y_train)

# print_results(cv)

# cv.best_estimator_

# joblib.dump(cv.best_estimator_,f'SVM_model_new_{Functions}.pkl')

#%% Machine Learning

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
import scikitplot as skplt
import matplotlib.pyplot as plt

#SVM
SVM1 = SVC()
scores = cross_val_score(SVM1, X_train, y_train, cv= 2)
AC=round(scores.mean(),3)
stds=round(scores.std(),3)
print(AC,stds)

SVM1.fit(X_train, y_train)
loaded_model = SVM1

# #kNN
# KNN1 = KNeighborsClassifier()
# scores = cross_val_score(KNN1, X_train, y_train, cv= 2)
# AC=round(scores.mean(),3)
# stds=round(scores.std(),3)
# print(AC,stds)

# KNN1.fit(X_train, y_train)
# loaded_model = KNN1

# #LR
# LR1 = LogisticRegression()
# scores = cross_val_score(LR1, X_train, y_train, cv= 2)
# AC=round(scores.mean(),3)
# stds=round(scores.std(),3)
# print(AC,stds)

# LR1.fit(X_train, y_train)
# loaded_model = LR1

# from sklearn.externals import joblib
# import joblib as jb
# loaded_model = jb.load(f'SVM_model_new_{Functions}.pkl')

y_pred_tr = loaded_model.predict(X_train)
skplt.metrics.plot_confusion_matrix(y_train,y_pred_tr, title= 'Train', cmap= 'Blues') 
tr_acc = round(accuracy_score(y_train,y_pred_tr), 3)
print(classification_report(y_train,y_pred_tr))

# The input will be X_val and it was x_val. both of them have different size 
y_pred_v = loaded_model.predict(X_val)
skplt.metrics.plot_confusion_matrix(y_val,y_pred_v, title= 'Validation', cmap= 'Greens') 
v_acc = round(accuracy_score(y_val,y_pred_v), 3)
print(classification_report(y_val,y_pred_v))

y_pred_ts = loaded_model.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test,y_pred_ts, title= 'Test', cmap= 'Oranges') 
ts_acc = round(accuracy_score(y_test,y_pred_ts),3)
print(classification_report(y_test,y_pred_ts))