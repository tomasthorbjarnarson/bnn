import numpy as np
import random

seed = 0

def numpy_sign(varMatrix):
  signVarMatrix = varMatrix
  signVarMatrix[varMatrix >= 0] = 1
  signVarMatrix[varMatrix < 0] = -1
  return signVarMatrix

def inference(set_x, varMatrices, architecture):
  N_test, input_size = np.shape(set_x)
    
  infer = set_x

  for lastLayer, neurons_out in enumerate(architecture[1:]):
    layer = lastLayer + 1

    infer = np.dot(np.transpose(varMatrices["w_%s" % layer]),infer)
    infer += np.reshape(varMatrices["b_%s" % layer], (neurons_out,1))
    if layer < len(architecture) - 1:
      infer = numpy_sign(infer)

  output = all_ok(infer)
  #output = all_good(infer)
  return output

def all_good(infer):
  output = []
  for row in np.transpose(infer):
    label = np.argwhere(row >= 0)
    if len(label) == 1 and len(label[0]) == 1:
      label = label[0][0]
    else:
      label = -1
    output.append(label)
  return output

def all_ok(infer):
  random.seed(seed)
  output = []
  for row in np.transpose(infer):
    label = np.argwhere(row == np.max(row))
    label = random.choice(label)[0]
    output.append(label)
  return output

def calc_accuracy(inferred, y):
  acc = 0
  for i, label in enumerate(inferred):
    if label == y[i]:
      acc += 1
  acc = acc/len(y)
  return acc*100

def clear_print(text):
  print("====================================")
  print(text)
  print("====================================")