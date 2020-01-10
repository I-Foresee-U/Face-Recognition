import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from data_processing import load_data, sample, separate

def PCA():
    train_img, train_label, test_img, test_label = load_data()
    # I choose No.1 folder to No.20 folder as class 1 to class 20
    # And set the 10 photos of mine as class 21

    subset_img, subset_label = sample(train_img, train_label)
    # the subset is the 500 samples that randomly sampled from the whole training set.

    my_subset_img, my_subset_label, PIE_subset_img, PIE_subset_label = separate(subset_img, subset_label)
    scatter_subset = np.cov(subset_img.T)
    U_subset, D_subset, Vt_subset = np.linalg.svd(scatter_subset, full_matrices=True)

    # Dimensionality reduction to 2D
    dimens_2 = U_subset[:,0:2]
    PIE_subset_2d = np.dot(PIE_subset_img, dimens_2)
    my_subset_2d = np.dot(my_subset_img,dimens_2)

    fig_2d = plt.figure()
    plot_2d = fig_2d.add_subplot(111)
    plot_2d.set_title('2D plot')
    plot_2d.scatter(PIE_subset_2d[:,0], PIE_subset_2d[:,1], c=PIE_subset_label, marker='.', label="PIE")
    if len(my_subset_label) == 1:
        plot_2d.scatter(my_subset_2d[0], my_subset_2d[1], c='r', marker='*', label="My Photo")
    else:
        plot_2d.scatter(my_subset_2d[:,0], my_subset_2d[:,1], c='r', marker='*', label="My Photo")
    plot_2d.legend()

    # Dimensionality reduction to 3D
    dimens_3 = U_subset[:, 0:3]
    my_subset_3d = np.dot(my_subset_img, dimens_3)
    PIE_subset_3d = np.dot(PIE_subset_img, dimens_3)

    fig_3d = plt.figure()
    plot_3d = fig_3d.add_subplot(111,projection='3d')
    plot_3d.set_title('3D plot')
    plot_3d.scatter(PIE_subset_3d[:,0], PIE_subset_3d[:,1], PIE_subset_3d[:,2], c=PIE_subset_label, marker='.', label="PIE")
    if len(my_subset_label) == 1:
        plot_3d.scatter(my_subset_3d[0], my_subset_3d[1], my_subset_3d[2], c='r', marker='*', label="My Photo")
    else:
        plot_3d.scatter(my_subset_3d[:,0], my_subset_3d[:,1], my_subset_3d[:,2], c='r', marker='*', label="My Photo")
    plot_3d.legend()

    # Visualize the corresponding 3 eigenfaces
    for i in range(3):
        eigenface = dimens_3[:, i].reshape((32,32))
        plt.figure()
        plt.imshow(eigenface, cmap=plt.cm.gray)
    plt.show()

    # Classification for test images
    my_test_img, my_test_label, PIE_test_img, PIE_test_label = separate(test_img, test_label)
    scatter_train = np.cov(train_img.T)
    U_train, D_train, Vt_train = np.linalg.svd(scatter_train, full_matrices=True)
    num_dimens = [40, 80, 200]
    for i in num_dimens:
        print('Dimensionality Reduction to',str(i),':')
        dimens = U_train[:,0:i]
        train_dimens = np.dot(train_img, dimens)
        PIE_test_dimens = np.dot(PIE_test_img, dimens)
        my_test_dimens = np.dot(my_test_img, dimens)
        acc_PIE = accuracy(train_dimens, train_label, PIE_test_dimens, PIE_test_label)
        acc_my = accuracy(train_dimens, train_label, my_test_dimens, my_test_label)
        print('Accuracy of PIE test images:',acc_PIE)
        print('Accuracy of my own photos:',acc_my)

def accuracy(train_X, train_y, test_X, test_y):
    W = train_X.shape[0]
    H = test_X.shape[0]
    qua_tr = np.square(train_X).sum(axis=1).reshape(1,W)
    qua_tr = np.tile(qua_tr,(H,1))
    qua_te = np.square(test_X).sum(axis=1).reshape(H,1)
    qua_te = np.tile(qua_te,(1,W))
    cross = np.dot(test_X, train_X.T)
    dist = np.add(qua_tr,qua_te)
    dist = np.add(dist, np.dot(-2,cross))
    dist = np.sqrt(np.clip(dist, 0, 1e10))
    correct = 0
    for i in range(H):
        neighbors = sorted(enumerate(dist[i]), key=lambda x:x[1])
        nearest_neighbor = neighbors[0][0]
        correct += (test_y[i] == train_y[nearest_neighbor])
    acc = correct/H

    return acc

if __name__ == "__main__":
    PCA()
