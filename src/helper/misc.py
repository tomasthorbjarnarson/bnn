import numpy as np
import random

seed = 0

def infer_and_accuracy(set_x, set_y, varMatrices, architecture):
  inferred = inference(set_x, varMatrices, architecture)
  accuracy = calc_accuracy(inferred, set_y)
  return accuracy

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
    infer = np.dot(infer,varMatrices["w_%s" % layer])
    infer += varMatrices["b_%s" % layer]
    if layer < len(architecture) - 1:
      infer = numpy_sign(infer)

  output = all_ok(infer)
  #output = all_good(infer)
  return output

def all_good(infer):
  output = []
  for row in infer:
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
  for row in infer:
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

def get_bound_matrix(network_vars, bound):
  """network_vars contains varMatrices of all batches"""
  all_vars = {}
  bound_matrix = {}
  for key in network_vars[0]:
    all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
    if "w_" in key or "b_" in key:
      tmp_min = np.min(all_vars[key],axis=0)
      tmp_max = np.max(all_vars[key],axis=0)
      tmp_min[tmp_min >= 0] = -bound
      tmp_max[tmp_max <= 0] = bound
      bound_matrix["%s_%s" % (key,"lb")] = tmp_min
      bound_matrix["%s_%s" % (key,"ub")] = tmp_max

  return bound_matrix

def get_alt_bound_matrix(network_vars, bound):
  """network_vars contains varMatrices of all batches"""
  all_vars = {}
  bound_matrix = {}
  for key in network_vars[0]:
    all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
    if "w_" in key or "b_" in key:
      bound_matrix["%s_%s" % (key,"lb")] = np.zeros_like(all_vars[key][0]) - bound
      bound_matrix["%s_%s" % (key,"ub")] = np.zeros_like(all_vars[key][0]) + bound
      vars_sum = all_vars[key].sum(axis=0) / all_vars[key].shape[0]
      vars_eq = np.equal(vars_sum, all_vars[key][0])
      bound_matrix["%s_%s" % (key,"lb")][vars_eq] = all_vars[key][0][vars_eq]
      bound_matrix["%s_%s" % (key,"ub")][vars_eq] = all_vars[key][0][vars_eq]

  return bound_matrix

def get_mean_vars(network_vars):
  """network_vars contains varMatrices of all batches"""
  all_vars = {}
  mean_vars = {}
  for key in network_vars[0]:
    all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
    mean_vars[key] = np.mean(all_vars[key], axis=0)
    mean_vars[key][mean_vars[key] < 0] -= 1e-5
    mean_vars[key][mean_vars[key] >= 0] += 1e-5
    mean_vars[key] = np.round(mean_vars[key])

  return mean_vars

def clear_print(text):
  print("====================================")
  print(text)
  print("====================================")