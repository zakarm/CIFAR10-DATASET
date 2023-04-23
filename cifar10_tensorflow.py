# /$$$$$$                      /$$ /$$          
# /$$__  $$                    | $$| $$          
#| $$  \ $$  /$$$$$$   /$$$$$$ | $$| $$  /$$$$$$ 
#| $$$$$$$$ /$$__  $$ /$$__  $$| $$| $$ /$$__  $$
#| $$__  $$| $$  \ $$| $$  \ $$| $$| $$| $$  \ $$
#| $$  | $$| $$  | $$| $$  | $$| $$| $$| $$  | $$
#| $$  | $$| $$$$$$$/|  $$$$$$/| $$| $$|  $$$$$$/
#|__/  |__/| $$____/  \______/ |__/|__/ \______/ 
#          | $$                                  
#          | $$                                  
#          |__/            BY Zakariae MRABET             

import tensorflow as tf 
import pandas as pd 
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def main():
    # load CIFAR10 dataset 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    """
        Normalize pixel values to be between 0 and 1Normalize pixel values to be between 0 and 1
    """
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # print head
    print("----------- Train Data -----------")
    print(x_train[:1])
    print("----------- Test Data -----------")
    print(x_test[:1])

    """
        This plots a 5x5 grid of the first 25 images in the 
        training set, along with their corresponding labels.
    """
    y_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
        plt.xlabel(y_names[y_train[i][0]])
    plt.show()

    """
        This defines the CNN architecture using 
        tf.keras.models.Sequential(), adding convolutional 
        layers and pooling layers. It also prints a summary 
        of the model's layers and parameters.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()

    """
        This defines the CNN architecture 
        using tf.keras.models.Sequential(), 
        adding convolutional layers and pooling layers. 
        It also prints a summary 
        of the model's layers and parameters.

        This adds fully connected layers to 
        the model and prints a summary of the updated model.
    """

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.summary()

    """
        This compiles the model, specifying the 
        Adam optimizer, sparse categorical cross-entropy loss, 
        and accuracy as the evaluation metric.
    """
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    """
        This trains the model on the training set for 10 epochs, 
        validating on the test set, 
        and saves the training history in the history variable.
    """
    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))

    """
        This plots the training and validation accuracy over epochs.
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

   
    """
        This evaluates the final performance of the model on the
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)


if __name__ == "__main__":
    main()
