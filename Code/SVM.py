import numpy as np
from svmutil import *

from data_processing import load_data

def SVM():
    train_img, train_label, test_img, test_label = load_data()
    # I choose No.1 folder to No.20 folder as class 1 to class 20
    # And set the 10 photos of mine as class 21

    scatter_train = np.cov(train_img.T)
    U_train, D_train, Vt_train = np.linalg.svd(scatter_train, full_matrices=True)
    num_dimens = [80, 200, 'raw']
    for i in num_dimens:
        if i == 'raw':
            print('Using raw face images:')
            train_dimens = train_img
            test_dimens = test_img
        else:
            print('After PCA pre-processing with dimensionality of', str(i), ':')
            dimens = U_train[:,0:i]
            train_dimens = np.dot(train_img, dimens)
            test_dimens = np.dot(test_img, dimens)
        C = ['0.00001', '0.01', '0.1', '1', '10000']
        for j in C:
            print('Penalty parameter C =', str(j), ':')
            model = svm_train(train_label, train_dimens, '-t 0 -c ' + j + ' -q')
            print('     Training images: ',end='')
            svm_predict(train_label, train_dimens, model)
            print('     Test images: ', end='')
            svm_predict(test_label, test_dimens, model)
        print('\n')

if __name__ == "__main__":
    SVM()