import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage.transform import resize
from skimage.filters import threshold_mean

#Kaggle Data set import - I had started this before the assignment was updated
dftr = pd.read_csv("C:/Users/Splinta/GoogleDrive/Masters/Courses/CS498/Assignments/HW1 P2/train.csv")
dfte = pd.read_csv("C:/Users/Splinta/GoogleDrive/Masters/Courses/CS498/Assignments/HW1 P2/test.csv")
#seperate label and features
trax = dftr.loc[:, dftr.columns != 'label']
tray = dftr.loc[:, dftr.columns == 'label']

#this makes no difference to threshold at 128
#trax[trax >= 128 ] = 255
#trax[trax < 128 ] = 0


#train test split - 80% train
X_train, X_test, Y_train, Y_test = train_test_split(trax, tray, test_size=0.2)



#Part A########################################################
#orignal image Naive Bayes Gaussian and Bernoulli
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
print("Part A NB Gaussian Original: \n")
print(classification_report(Y_test, y_pred))

bnb = BernoulliNB()
y_predb = bnb.fit(X_train, Y_train).predict(X_test)
print("Part A NB Bernoulli Original: \n")
print(classification_report(Y_test, y_predb))


#Streched Bounded Box image
#Reshape to 28x28 image
trainrs = np.array(X_train).reshape((-1, 1, 28, 28)).astype(np.uint8)
testrs = np.array(X_test).reshape((-1, 1, 28, 28)).astype(np.uint8)

#https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
#function to crop image to closest ink pixels on vertical and horizontal axis
#it also accepts a tolerance/threshold value
def crop_image(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

#train threshold
#Take the mean threshold of an image and crop image using the threshold
#then resize the croped image to 20x20 and insert into array
trainarray = np.zeros(shape=(trainrs.shape[0],20,20))
for i in range(0,trainrs.shape[0]):
    thresh_mean = threshold_mean(trainrs[i][0])
    loopcrop = (crop_image(trainrs[i][0],thresh_mean))
    newimg = resize(loopcrop, (20, 20),mode="constant")
    trainarray[i] = newimg


#test threshold
#Take the mean threshold of an image and crop image using the threshold
#then resize the croped image to 20x20 and insert into array
testarray = np.zeros(shape=(testrs.shape[0],20,20))
for i in range(0,testrs.shape[0]):
    thresh_mean = threshold_mean(testrs[i][0])
    loopcrop = (crop_image(testrs[i][0],thresh_mean))
    newimg = resize(loopcrop, (20, 20),mode="constant")
    testarray[i] = newimg


#testing the image quality, orignal v.s. stretched and bounded image
#plt.imshow(trainrs[30][0], cmap=cm.binary) # draw the picture
plt.imshow(trainarray[30], cmap=cm.binary) # draw the picture

#plt.imshow(testrs[1780][0], cmap=cm.binary) # draw the picture
#plt.imshow(testarray[1780], cmap=cm.binary) # draw the picture


#undo 28x28 transformation and return the pixels to a single row for classification
trainfa = trainarray.reshape((33600, -1))
testfa = testarray.reshape((8400, -1))

#do Gaussian and Bernoulli NB on the transformed stretched and bounded image
#significant improvement
gnb = GaussianNB()
y_predcb = gnb.fit(trainfa, Y_train).predict(testfa)
print("Part A NB Gaussian Stretched: \n")
print(classification_report(Y_test, y_predcb))

bnb = BernoulliNB()
y_predcbb = bnb.fit(trainfa, Y_train).predict(testfa)
print("Part A NB Bernoulli Stretched: \n")
print(classification_report(Y_test, y_predcbb))

#Part B################################################################
#######################################################################
#re pulling ALL of the training data and transforming. then doing cross val times 5
# so that the data is split 80% trian and 20% test
#same logic as above

X_train = trax
Y_train = tray
trainrs = np.array(X_train).reshape((-1, 1, 28, 28)).astype(np.uint8)

trainarray = np.zeros(shape=(trainrs.shape[0],20,20))
for i in range(0,trainrs.shape[0]):
    thresh_mean = threshold_mean(trainrs[i][0])
    loopcrop = (crop_image(trainrs[i][0],thresh_mean))
    newimg = resize(loopcrop, (20, 20),mode="constant")
    trainarray[i] = newimg

trainfa = trainarray.reshape((42000, -1))

clf = RandomForestClassifier(random_state=0)

#stretched and bounded image using Decision Forests
#did Grid search and CV using the parameters provided in the assignment
param_grid = [{'max_depth': [4, 8, 16], 'n_estimators': [10, 20, 30]}]
gsclf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
gsclf.fit(trainfa,np.ravel(Y_train))
gscvresults = pd.DataFrame(dict(gsclf.cv_results_))
print("Part B Decision Forests Stretched: \n",
      pd.concat([gscvresults.iloc[:,2],gscvresults.iloc[:,6]],axis=1))

#original image using Decision Forests
#did Grid search and CV using the parameters provided in the assignment
param_grid = [{'max_depth': [4, 8, 16], 'n_estimators': [10, 20, 30]}]
gsclf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
gsclf.fit(X_train,np.ravel(Y_train))
gscvresultso = pd.DataFrame(dict(gsclf.cv_results_))
print("Part B Decision Forests Original: \n",
      pd.concat([gscvresultso.iloc[:,2],gscvresultso.iloc[:,6]],axis=1))
