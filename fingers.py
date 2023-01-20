import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.math import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

"""
Source code inspired and modified from:
https://www.kaggle.com/code/esraaashraf99/fingers-detection-cnn-tensorflow-keras
"""

DATADIR_Train = './fingers/train'
DATADIR_Test = './fingers/test'

IMG_SIZE = 128
label_list = ['0L', '1L', '2L', '3L', '4L', '5L', '0R', '1R', '2R', '3R', '4R', '5R']

training_data = []
test_data = []

def Grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def GaussianBlur(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def SobelFilter(image):
    image = Grayscale(GaussianBlur(image))
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    size = image.shape
    kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            G_x[i, j] = np.sum(np.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_x))
            G_y[i, j] = np.sum(np.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_y))

    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles


def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed


def double_threshold_hysteresis(image, low, high):
    weak = 50
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape

    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if ((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y] == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result


def Canny(image, low, high):
    image, angles = SobelFilter(image)
    image = non_maximum_suppression(image, angles)
    gradient = np.copy(image)
    image = double_threshold_hysteresis(image, low, high)
    return image, gradient

def preprocess(img):
    image = cv2.resize(img, (128, 128))
    image, gradient = Canny(image, 0, 50)
    plt.imshow(image,cmap='gray')
    plt.show()
    input = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    input = input/255.0
    return input

def run():
    c=0
    for train_img in os.listdir(DATADIR_Train):
        image = cv2.imread(os.path.join(DATADIR_Train, train_img))
        image, gradient = Canny(image, 0, 50)
        label_str = train_img[-5: -7: -1][::-1]
        label = label_list.index(label_str)
        #print(label_str)
        #plt.imshow(image,cmap='gray')
        training_data.append([image, label])
        #plt.show()
        c=c+1
        print(c)
    x = 0
    for test_img in os.listdir(DATADIR_Test):
        image = cv2.imread(os.path.join(DATADIR_Test, test_img))
        image, gradient = Canny(image, 0, 50)
        label_str_test = test_img[-5: -7: -1][::-1]
        label_test = label_list.index(label_str_test)
        #print(label_str_test)
        #plt.imshow(image,cmap='gray')
        test_data.append([image, label_test])
        #plt.show()
        x = x + 1
        print(x)


    random.shuffle(training_data)
    random.shuffle(test_data)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for feature, label in training_data:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test_data:
        x_test.append(feature)
        y_test.append(label)

    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train/255.0
    x_test = x_test/255.0

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(12))
    model.add(Activation('softmax'))

    adam_optimizer = Adam(learning_rate=0.001)

    model.compile(
    optimizer=adam_optimizer,
    loss=sparse_categorical_crossentropy,
    metrics=['accuracy']
    )

    training_history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test))

    model.save('fingerCNN.h5', save_format='h5')


    #model = load_model("fingerCNN.h5")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    confusion_mx = confusion_matrix(y_test, y_pred)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mx, display_labels=label_list)

    cm_display.plot()
    plt.show()
