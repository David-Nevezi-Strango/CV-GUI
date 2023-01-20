
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

"""
Source code inspired and modified from:
https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy
"""

def preprocess(img):
    image = cv2.resize(img, (28,28))
    plt.imshow(image,cmap='gray')
    plt.show()
    input = np.array(image).reshape(-1,28,28,1)
    input = input / 255.0
    return input

def run():
    train_df = pd.read_csv("handsign/sign_mnist_train/sign_mnist_train.csv")
    test_df = pd.read_csv("handsign/sign_mnist_test/sign_mnist_test.csv")

    y_train = train_df['label']
    y_test = test_df['label']
    del train_df['label']
    del test_df['label']

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)
    print(label_binarizer.inverse_transform(y_train))
    x_train = train_df.values
    x_test = test_df.values

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)


    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range = 0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)


    datagen.fit(x_train)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

    model = Sequential()
    model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    model.fit(datagen.flow(x_train,y_train, batch_size = 128) ,epochs = 20 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])
    
    model.save("handsign.h5", save_format="h5")
    model = load_model("handsign.h5")
    #print("acc " , model.evaluate(x_test,y_test)[1])

    predictions = np.argmax(model.predict(x_test), axis=1)
    for i in range(len(predictions)):
        if(predictions[i] >= 9):
            predictions[i] += 1
    print(predictions)

