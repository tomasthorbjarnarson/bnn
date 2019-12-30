import numpy as np
from mlxtend.data import loadlocal_mnist
from mlxtend.preprocessing import one_hot
import matplotlib.pyplot as plt

def preprocess_mnist(set_x):
  proc_x = set_x/255
  proc_x = np.transpose(proc_x)
  return proc_x

def load_mnist(N, ):

  train_x, train_y = loadlocal_mnist(
          images_path='data/raw/train-images-idx3-ubyte', 
          labels_path='data/raw/train-labels-idx1-ubyte')

  test_x, test_y = loadlocal_mnist(
          images_path='data/raw/t10k-images-idx3-ubyte', 
          labels_path='data/raw/t10k-labels-idx1-ubyte')

  train_x = preprocess_mnist(train_x[0:N])

  train_y = train_y[0:N]
  oh_train_y = one_hot(train_y, num_labels=10)
  oh_train_y = np.transpose(oh_train_y)

  test_x = preprocess_mnist(test_x)
  oh_test_y = one_hot(test_y, num_labels=10)
  oh_test_y = np.transpose(oh_test_y)

  return train_x, train_y, oh_train_y, test_x, test_y, oh_test_y


def imshow(example):
  plt.imshow(example.reshape([28, 28]))
  plt.show()