import numpy as np
import random
from PIL import Image
import os

def load_data():
    train_img = []
    train_label = []
    test_img = []
    test_label = []
    for i in range(21):
        path = os.path.join('./PIE',str(i+1))
        name_list = os.listdir(path)
        test_list = random.sample(name_list,int(len(name_list)*0.3))
        train_list = list(set(name_list).difference(set(test_list)))
        for name in train_list:
            img = Image.open(os.path.join(path,str(name)))
            img_vec = np.array(img).reshape(1,1024)
            if train_img == []:
                train_img = img_vec
            else:
                train_img = np.vstack((train_img, img_vec))
            train_label.append(i+1)
        for name in test_list:
            img = Image.open(os.path.join(path, str(name)))
            img_vec = np.array(img).reshape(1,1024)
            if test_img == []:
                test_img = img_vec
            else:
                test_img = np.vstack((test_img, img_vec))
            test_label.append(i+1)

    return train_img, train_label, test_img, test_label

def sample(img, label):
    while True:
        subset_list = random.sample(range(img.shape[0]), 500)
        subset_img = []
        subset_label = []
        num = 0
        for i in subset_list:
            subset_label.append(label[i])
            if label[i] == 21:
                num += 1
            if subset_img == []:
                subset_img = img[i]
            else:
                subset_img = np.vstack((subset_img, img[i]))
        if num != 0:
            break

    return subset_img, subset_label

def separate(img, label):
    my_img = []
    my_label = []
    PIE_img = []
    PIE_label = []
    for i in range(len(label)):
        if label[i] == 21:
            my_label.append(label[i])
            if my_img == []:
                my_img = img[i]
            else:
                my_img = np.vstack((my_img, img[i]))
        else:
            PIE_label.append(label[i])
            if PIE_img == []:
                PIE_img = img[i]
            else:
                PIE_img = np.vstack((PIE_img, img[i]))

    return my_img, my_label, PIE_img, PIE_label
