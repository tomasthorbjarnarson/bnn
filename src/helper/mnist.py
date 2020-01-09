from mlxtend.data import loadlocal_mnist
from mlxtend.preprocessing import one_hot
import matplotlib.pyplot as plt
import numpy as np
import math

def preprocess_mnist(set_x):
  proc_x = set_x/255
  proc_x = np.transpose(proc_x)
  return proc_x

def get_unique_examples(train_x, train_y, N):
  ex_per_class = math.ceil(N / 10)
  indices = []
  for i, ex in enumerate(train_x):
    if len(indices) < ex_per_class*10 and np.count_nonzero(train_y[indices] == train_y[i]) < ex_per_class:
      indices.append(i)

  return indices[0:N]

def load_mnist(N):

  og_train_x, og_train_y = loadlocal_mnist(
          images_path='data/raw/train-images-idx3-ubyte', 
          labels_path='data/raw/train-labels-idx1-ubyte')

  test_x, test_y = loadlocal_mnist(
          images_path='data/raw/t10k-images-idx3-ubyte', 
          labels_path='data/raw/t10k-labels-idx1-ubyte')

  max_num = len(og_train_y)

  og_train_x = preprocess_mnist(og_train_x)

  og_oh_train_y = one_hot(og_train_y, num_labels=10)
  og_oh_train_y = np.transpose(og_oh_train_y)

  train_indices = get_unique_examples(og_train_x, og_train_y, N)
  val_indices = [i for i in range(max_num) if i not in train_indices]

  train_x = og_train_x[:,train_indices]
  train_y = og_train_y[train_indices]
  oh_train_y = og_oh_train_y[:,train_indices]

  val_x = og_train_x[:,val_indices]
  val_y = og_train_y[val_indices]
  oh_val_y = og_oh_train_y[:,val_indices]

  test_x = preprocess_mnist(test_x)
  oh_test_y = one_hot(test_y, num_labels=10)
  oh_test_y = np.transpose(oh_test_y)

  return train_x, train_y, oh_train_y, val_x, val_y, oh_val_y, test_x, test_y, oh_test_y


def imshow(example):
  plt.imshow(example.reshape([28, 28]))
  plt.show()