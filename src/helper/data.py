from mlxtend.data import loadlocal_mnist
from mlxtend.preprocessing import one_hot
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def preprocess_mnist(set_x):
  # Normalize pixels to be between 0 and 1
  proc_x = set_x/255
  # Use this to simply set to float
  #proc_x = set_x / 1.0
  proc_x = np.transpose(proc_x)
  return proc_x

def get_unique_examples(train_x, train_y, N):
  ex_per_class = math.ceil(N / 10)
  indices = []
  for i, ex in enumerate(train_x):
    if len(indices) < ex_per_class*10 and np.count_nonzero(train_y[indices] == train_y[i]) < ex_per_class:
      indices.append(i)

  return indices[0:N]

def load_mnist(N, seed):

  random.seed(seed)

  og_train_x, og_train_y = loadlocal_mnist(
          images_path='data/raw/train-images-idx3-ubyte', 
          labels_path='data/raw/train-labels-idx1-ubyte')

  test_x, test_y = loadlocal_mnist(
          images_path='data/raw/t10k-images-idx3-ubyte', 
          labels_path='data/raw/t10k-labels-idx1-ubyte')

  og_train_indices = [x for x in range(len(og_train_x))]
  if seed:
    random.shuffle(og_train_indices)

  og_train_x = og_train_x[og_train_indices]
  og_train_y = og_train_y[og_train_indices]

  max_num = len(og_train_y)

  og_train_x = preprocess_mnist(og_train_x)
  # Map labels to 1 and -1 instead of 1 and 0
  og_oh_train_y = 2*one_hot(og_train_y, num_labels=10)-1
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
  oh_test_y = 2*one_hot(test_y, num_labels=10)-1
  oh_test_y = np.transpose(oh_test_y)

  data = {
    "train_x": train_x,
    "train_y": train_y,
    "oh_train_y": oh_train_y,
    "val_x": val_x,
    "val_y": val_y,
    "oh_val_y": oh_val_y,
    "test_x": test_x,
    "test_y": test_y,
    "oh_test_y": oh_test_y,
  }

  return data

def oh_encode(labels):
  classes = np.max(labels) 
  encoded = np.zeros((len(labels), classes+1)) - 1

  for i,val in enumerate(labels):
    encoded[i][val] = 1
  return encoded

def load_data(dataset, N, seed):
  random.seed(seed)

  (train_x, train_y), (test_x, test_y) = dataset.load_data()
  # Normalize
  train_x = train_x/255
  test_x = test_x/255

  pixels = np.prod(train_x.shape[1:])
  train_x = train_x.reshape(-1, pixels)
  test_x = test_x.reshape(-1, pixels)

  train_indices = [x for x in range(len(train_x))]

  if seed:
    random.shuffle(train_indices)

  train_x = train_x[train_indices]
  train_y = train_y[train_indices]

  max_num = len(train_x)
  train_indices = get_unique_examples(train_x, train_y, N)
  val_indices = [i for i in range(max_num) if i not in train_indices]

  data = {}
  data["train_x"] = train_x[train_indices]
  data["train_y"] = train_y[train_indices]
  data["oh_train_y"] = oh_encode(train_y[train_indices])

  data["val_x"] = train_x[val_indices]
  data["val_y"] = train_y[val_indices]
  data["oh_val_y"] = oh_encode(train_y[val_indices])

  data["test_x"] = test_x
  data["test_y"] = test_y
  data["oh_val_y"] = oh_encode(test_y)

  return data


def imshow(example):
  plt.imshow(example.reshape([28, 28]))
  plt.show()