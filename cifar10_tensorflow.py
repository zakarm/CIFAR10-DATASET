import tensorflow as tf 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def main():
    # load CIFAR10 dataset 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # norm data between 0 and 1 
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # print head
    print("----------- Train Data -----------")
    print(x_train[:1])
    print("----------- Test Data -----------")
    print(x_test[:1])

    # show images 
    y_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(10, 10))
    for i in range (25): 
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
        plt.xlabel(y_names[y_train[i][0]])
    plt.show()

    


if __name__ == "__main__":
    main()