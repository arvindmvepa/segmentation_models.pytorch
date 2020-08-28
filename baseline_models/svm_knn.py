#As other classifiers, SVC, NuSVC and LinearSVC take as input two arrays:
#an array X of size [n_samples, n_features] holding the training samples,
#and an array y of class labels (strings or integers), size [n_samples]:

from sklearn import svm
from PIL import Image
import os
import numpy as np
from numpy  import *
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


# TO DO: still need to look at

# feature vector
def feature_vector(dataset = None , testflag = None):

    dataset = dataset
    file_dir = "/root/vessel-seg/data/{}/{}".format(dataset,testflag)
    img_dir = os.path.join(file_dir, "images")
    target_dir = os.path.join(file_dir, "targets")
    img_names = [fname for fname in os.listdir(img_dir) if fname.endswith(".jpg")]
    img_names = sorted(img_names)
    img_num = len(img_names)

    img = Image.open("{}/{}".format(img_dir,img_names[0]))
    img_arr = np.asarray(img)
    img_shape = img_arr.shape

    feature_vector = np.zeros((img_num * img_shape[0] * img_shape[1],31))

    for i in range(img_num):
        feature_dir = os.path.join(file_dir, "features",img_names[i][:-4])
        feature_names = [fname for fname in os.listdir(feature_dir) if fname.endswith(".jpg")]
        feature_names = sorted(feature_names)
        feature_arr = np.zeros((31,img_shape[0],img_shape[1]))

        for file_index in range(31):
            img = Image.open("{}/{}".format(feature_dir,feature_names[file_index]))
            feature_arr[file_index,:,:] = np.asarray(img)


        for height in range(img_shape[0]):
            for weight in range(img_shape[1]):
                feature_index = i*img_shape[0]*img_shape[1]+height*img_shape[1]+weight
                feature_vector[feature_index,:] = feature_arr[:,height,weight]

    if testflag == 'test':
        print("img_num",img_num)
        print("test_feature_vector",feature_vector.shape)
        return feature_vector,img_num

    target_names = [fname for fname in os.listdir(target_dir) if fname.endswith(".png")]
    target_names = sorted(target_names)
    target_arr = np.zeros((img_num,img_shape[0]*img_shape[1]))

    for i in range(len(target_names)):
        target = np.asarray(Image.open("{}/{}".format(target_dir,target_names[i]))).flatten()
        target_arr[i,:] = target
    target_arr = target_arr.flatten()
    target_arr = target_arr.astype(int)


    print("img_num",img_num)
    print("train_feature_vector",feature_vector.shape)
    print("target_arr",target_arr.shape)

    return feature_vector,target_arr,img_shape

train_feature_arr,target_arr,img_size = feature_vector('CHASE','train')
print(img_size)
test_feature_arr, img_num = feature_vector('CHASE','test')

#use svm
'''
clf = svm.SVC()
clf.fit(train_feature_arr, target_arr)
'''

#use knn
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_feature_arr, target_arr)

joblib.dump(neigh, 'filename.pkl')
test_predict = neigh.predict(test_feature_arr)
test_predict = test_predict.reshape(img_size[0]*img_size[1],img_num)

for i in range(img_num):
    predict_img = test_prediction[:,i].reshape(img_size[0],img_size[1])
    predict_img = Image.fromarray(predict_img)
    predict_img.save("/root/vessel-seg/data/CHASE/test/predictions/{}.jpg".format(i))
