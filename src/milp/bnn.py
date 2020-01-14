"""
This work heavily relies on the implementation by Icarte:
https://bitbucket.org/RToroIcarte/bnn/src/master/
"""

import numpy as np
from helper.mnist import load_mnist
from globals import INT, BIN, CONT

class BNN:
  def __init__(self, model, N, architecture):

    self.N = N
    self.architecture = architecture
    self.train_x, self.train_y, self.oh_train_y, self.val_x, self.val_y, self.oh_val_y, self.test_x, self.test_y, self.oh_test_y = load_mnist(N)
    dead = np.all(self.train_x == 0, axis=1)

    self.weights = {}
    self.biases = {}
    self.var_c = {}
    self.act = {}

    self.abs_weights = {}
    self.abs_biases = {}

    p_bound = 1
    pre_bound = 1
    epsilon = 1

    self.m = model

    self.periodic = []

    for lastLayer, neurons_out in enumerate(architecture[1:]):
      layer = lastLayer + 1
      neurons_in = architecture[lastLayer]

      self.weights[layer] = np.full((neurons_in, neurons_out), None)
      self.abs_weights[layer] = np.full((neurons_in, neurons_out), None)
      self.biases[layer] = np.full(neurons_out, None)
      self.abs_biases[layer] = np.full(neurons_out, None)

      if layer > 1:
        self.var_c[layer] = np.full((N, neurons_in, neurons_out), None)
      if layer < len(architecture) - 1:
        self.act[layer] = np.full((N, neurons_out), None)

      for j in range(neurons_out):
        for i in range(neurons_in):
          if layer == 1 and dead[i]:
            # Dead inputs should have 0 weight and therefore 0 absolute weight
            self.weights[layer][i,j] = 0
            self.abs_weights[layer][i,j] = 0
          else:
            self.weights[layer][i,j] = self.add_var(INT,"w_%s-%s_%s" % (layer,i,j), p_bound)
            self.abs_weights[layer][i,j] = self.add_var(BIN,"abs(w)_%s-%s_%s" % (layer,i,j))
            self.add_constraint(self.abs_weights[layer][i,j] >= self.weights[layer][i,j])
            self.add_constraint(-self.abs_weights[layer][i,j] <= self.weights[layer][i,j])
          if layer > 1:
            # Var c only needed after first activation
            for k in range(N):
              self.var_c[layer][k,i,j] = self.add_var(CONT,"c_%s-%s_%s_%s" % (layer,i,j,k), pre_bound)
        # Bias only for each output neuron
        self.biases[layer][j] = self.add_var(INT,"b_%s-%s" % (layer,j), p_bound)
        self.abs_biases[layer][j] = self.add_var(BIN,"abs(b)_%s-%s" % (layer,j))
        self.add_constraint(self.abs_biases[layer][j] >= self.biases[layer][j])
        self.add_constraint(-self.abs_biases[layer][j] <= self.biases[layer][j])

        if layer < len(architecture) - 1:
          for k in range(N):
            self.act[layer][k,j] = self.add_var(BIN, name="act_%s-%s_%s" % (layer,j,k))


    for lastLayer, neurons_out in enumerate(architecture[1:]):
      layer = lastLayer + 1
      neurons_in = architecture[lastLayer]

      for k in range(N):
        for j in range(neurons_out):
          inputs = []
          for i in range(neurons_in):
            if layer == 1:
              inputs.append(self.train_x[i,k]*self.weights[layer][i,j])
            else:
              self.add_constraint(self.var_c[layer][k,i,j] - self.weights[layer][i,j] + 2*self.act[lastLayer][k,i] <= 2)
              self.add_constraint(self.var_c[layer][k,i,j] + self.weights[layer][i,j] - 2*self.act[lastLayer][k,i] <= 0)
              self.add_constraint(self.var_c[layer][k,i,j] - self.weights[layer][i,j] - 2*self.act[lastLayer][k,i] >= -2)
              self.add_constraint(self.var_c[layer][k,i,j] + self.weights[layer][i,j] + 2*self.act[lastLayer][k,i] >= 0)
              inputs.append(self.var_c[layer][k,i,j])
          pre_activation = sum(inputs) + self.biases[layer][j]

          if layer < len(architecture) - 1:
            self.add_constraint((self.act[layer][k,j] == 1) >> (pre_activation >= 0))
            self.add_constraint((self.act[layer][k,j] == 0) >> (pre_activation <= -epsilon))
          else:
            if self.oh_train_y[j,k] > 0:
              self.add_constraint(pre_activation >= 0)
            else:
              self.add_constraint(pre_activation <= -epsilon)

    self.objSum = 0
    for layer in self.abs_weights:
      self.objSum += self.abs_weights[layer].sum()
    for layer in self.abs_biases:
      self.objSum += self.abs_biases[layer].sum()

  def add_var(self, precision, name, bound=0):
    raise NotImplementedError("Add variable not implemented")

  def add_constraint(self, constraint):
    raise NotImplementedError("Add constraint not implemented")
    
  def train(self, time=None, focus=None):
    raise NotImplementedError("Train not implemented")

  def get_objective(self):
    raise NotImplementedError("Get objective not implemented")

  def get_runtime(self):
    raise NotImplementedError("Get runtime not implemented")
    
  def get_val(self, maybe_var):
    raise NotImplementedError("Get value not implemented")

  def extract_values(self):
    get_val = np.vectorize(self.get_val)

    varMatrices = {}
    for layer in self.weights:
      varMatrices["w_%s" %layer] = get_val(self.weights[layer])
      varMatrices["b_%s" %layer] = get_val(self.biases[layer])
      if layer > 1:
        varMatrices["c_%s" %layer] = get_val(self.var_c[layer])
      if layer < len(self.architecture) - 1:
        varMatrices["act_%s" %layer] = get_val(self.act[layer])

    return varMatrices


  def print_values(self):
    get_val = np.vectorize(self.get_val)

    for layer in self.weights:
      print("Weight %s" % layer)
      print(get_val(self.weights[layer]))
      print("Biases %s" % layer)
      print(get_val(self.biases[layer]))
      if layer > 1:
        print("C %s" % layer)
        print(get_val(self.var_c[layer]))
      if layer < len(self.architecture) - 1:
        print("Activations %s" % layer)
        print(get_val(self.act[layer]))
