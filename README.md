# CIFAR10-DATASET
## Introduction :

The CIFAR (Canadian Institute for Advanced Research) dataset is a collection of labeled images that is widely used in the field of computer vision for benchmarking machine learning algorithms. There are two main versions of the CIFAR dataset:

1. CIFAR-10: This dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

2. CIFAR-100: This dataset contains 60,000 32x32 color images in 100 classes, with 600 images per class. The classes are grouped into 20 superclasses, each containing 5 subclasses.

The CIFAR dataset is often used as a benchmark for image classification algorithms because of its relatively small size and simplicity compared to other datasets such as ImageNet. It is commonly used to evaluate the performance of deep learning models, particularly convolutional neural networks (CNNs).

The CIFAR dataset has been used in a wide range of research areas, including object recognition, image segmentation, image restoration, and generative models. Many machine learning frameworks, such as TensorFlow and PyTorch, provide built-in functions to download and load the CIFAR dataset for training and testing machine learning models.

<p align="center">
    <img src="https://github.com/zakarm/CIFAR10-DATASET/blob/main/cifar-10.png" width="300" height="300">
</p>

## Load CIFAR10-DATASET

There are several ways to download and install the CIFAR dataset, depending on your needs and the machine learning framework you are using. Here are a few options:

1. Using TensorFlow: If you are using TensorFlow, you can use the tf.keras.datasets.cifar10.load_data() function to download and load the CIFAR-10 dataset, or `tf.keras.datasets.cifar100.load_data()` function to download and load the CIFAR-100 dataset. These functions will automatically download the dataset and return it in a format that can be used for training and testing machine learning models.

2. Using PyTorch: If you are using PyTorch, you can use the `torchvision.datasets.CIFAR10` and `torchvision.datasets.CIFAR100` classes to download and load the CIFAR-10 and CIFAR-100 datasets, respectively. These classes will automatically download the dataset and return it in a format that can be used for training and testing machine learning models.

3. Downloading manually: If you prefer to download the dataset manually, you can visit the official CIFAR website at https://www.cs.toronto.edu/~kriz/cifar.html and download the dataset in either binary or text format. Once you have downloaded the dataset, you can use a script to convert the data into a format that can be used for machine learning.

Regardless of the method you choose, it's important to note that the CIFAR dataset is relatively large, and may take some time to download depending on your internet connection speed.

## Training convolutional neural network (CNN)

This is a Python code for training a convolutional neural network (CNN) on the CIFAR-10 dataset using TensorFlow Keras. Here is a summary of what the code does:

1. Load the CIFAR-10 dataset using tf.keras.datasets.cifar10.load_data().
2. Normalize the pixel values in the input images to be between 0 and 1.
3. Print the first image in the training and test sets, along with their corresponding labels.
4. Define a CNN architecture using tf.keras.models.Sequential() and add convolutional layers, pooling layers, and fully connected layers.
5. Compile the model using the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.
6. Train the model on the training set for 10 epochs and validate on the test set.
7. Plot the training and validation accuracy over the epochs.
8. Evaluate the final performance of the model on the test set by computing the test loss and accuracy using model.evaluate().
9. Note that the code uses matplotlib for visualizing the images and the training/validation accuracy over epochs. It also assumes that TensorFlow Keras and matplotlib libraries are installed.
