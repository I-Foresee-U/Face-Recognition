import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from data_processing import load_data

def CNN():
    train_img, train_label, test_img, test_label = load_data()
    # I choose No.1 folder to No.20 folder as class 1 to class 20
    # And set the 10 photos of mine as class 21

    train_X = train_img.reshape(train_img.shape[0], 32, 32, 1).astype('float32')/255.
    test_X = test_img.reshape(test_img.shape[0], 32, 32, 1).astype('float32')/255.
    train_y = to_categorical(np.array(train_label)-1, num_classes=21)
    test_y = to_categorical(np.array(test_label)-1, num_classes=21)

    model = Sequential()
    # The 1st CONV layer with 20 nodes
    model.add(Conv2D(filters=20, kernel_size=5, strides=1, padding="same", input_shape=(32, 32, 1)))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation('relu'))
    # The 2nd CONV layer with 50 nodes
    model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation('relu')) #use Relu as the activate function
    # The FC layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    # The Output layer with 21 nodes
    model.add(Dense(21))
    model.add(Activation('softmax'))
    adam = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    nn = model.fit(train_X, train_y, epochs = 30, batch_size = 256,
                   validation_data=(test_X, test_y),
                   shuffle=True, verbose=1, )

    # Evaluating the model
    loss_train, acc_train = model.evaluate(train_X, train_y, verbose=0)
    loss_test, acc_test = model.evaluate(test_X, test_y, verbose=0)
    print('Accuracy of training set:', acc_train, 'loss:', loss_train)
    print('Accuracy of testing set:', acc_test, 'loss:', loss_test)

if __name__ == "__main__":
    CNN()
