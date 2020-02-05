"""
This work heavily relies on the implementation by Icarte:
https://bitbucket.org/RToroIcarte/bnn/src/master/
"""

import numpy as np
from helper.mnist import load_data
from globals import INT, BIN, CONT, EPSILON

class BNN:
  def __init__(self, model, dataset, N, architecture, seed=0):

    self.N = N
    self.architecture = architecture
    data = load_data(dataset,N, seed)
    self.architecture[0] = data['train_x'].shape[1]
    self.data = data
    self.train_x = data["train_x"]
    self.oh_train_y = data["oh_train_y"]

    self.m = model

    self.init_params()
    self.add_examples()

  def init_params(self):
    self.weights = {}
    self.biases = {}
    self.var_c = {}
    self.act = {}

    self.bound = 1
    # All pixels that are 0 in every example are considered dead
    dead = np.all(self.train_x == 0, axis=0)

    for lastLayer, neurons_out in enumerate(self.architecture[1:]):
      layer = lastLayer + 1
      neurons_in = self.architecture[lastLayer]

      self.weights[layer] = np.full((neurons_in, neurons_out), None)
      self.biases[layer] = np.full(neurons_out, None)

      if layer > 1:
        self.var_c[layer] = np.full((self.N, neurons_in, neurons_out), None)
      if layer < len(self.architecture) - 1:
        self.act[layer] = np.full((self.N, neurons_out), None)

      for j in range(neurons_out):
        for i in range(neurons_in):
          if layer == 1 and dead[i]:
            # Dead inputs should have 0 weight
            self.weights[layer][i,j] = 0
          else:
            self.weights[layer][i,j] = self.add_var(INT,"w_%s-%s_%s" % (layer,i,j), self.bound)
          if layer > 1:
            # Var c only needed after first activation
            for k in range(self.N):
              self.var_c[layer][k,i,j] = self.add_var(CONT,"c_%s-%s_%s_%s" % (layer,i,j,k), self.bound)
        # Bias only for each output neuron
        self.biases[layer][j] = self.add_var(INT,"b_%s-%s" % (layer,j), self.bound)

        if layer < len(self.architecture) - 1:
          for k in range(self.N):
            # Each neuron for every example is either activated or not
            self.act[layer][k,j] = self.add_var(BIN, name="act_%s-%s_%s" % (layer,j,k))

  def add_examples(self):
    for lastLayer, neurons_out in enumerate(self.architecture[1:]):
      layer = lastLayer + 1
      neurons_in = self.architecture[lastLayer]

      for k in range(self.N):
        for j in range(neurons_out):
          inputs = []
          for i in range(neurons_in):
            if layer == 1:
              inputs.append(self.train_x[k,i]*self.weights[layer][i,j])
            else:
              self.add_constraint(self.var_c[layer][k,i,j] - self.weights[layer][i,j] + 2*self.bound*self.act[lastLayer][k,i] <= 2*self.bound)
              self.add_constraint(self.var_c[layer][k,i,j] + self.weights[layer][i,j] - 2*self.bound*self.act[lastLayer][k,i] <= 0*self.bound)
              self.add_constraint(self.var_c[layer][k,i,j] - self.weights[layer][i,j] - 2*self.bound*self.act[lastLayer][k,i] >= -2*self.bound)
              self.add_constraint(self.var_c[layer][k,i,j] + self.weights[layer][i,j] + 2*self.bound*self.act[lastLayer][k,i] >= 0*self.bound)
              inputs.append(self.var_c[layer][k,i,j])
          pre_activation = sum(inputs) + self.biases[layer][j]

          if layer < len(self.architecture) - 1:
            self.add_constraint((self.act[layer][k,j] == 1) >> (pre_activation >= 0))
            self.add_constraint((self.act[layer][k,j] == 0) >> (pre_activation <= -EPSILON))

  def add_output_constraints(self):
    raise NotImplementedError("Add output constraints not implemented")

  def calc_objective(self):
    raise NotImplementedError("Calculate objective not implemented")

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
      varMatrices["w_%s" %layer] = self.get_val(self.weights[layer])
      varMatrices["b_%s" %layer] = self.get_val(self.biases[layer])
      if layer > 1:
        varMatrices["c_%s" %layer] = self.get_val(self.var_c[layer])
      if layer < len(self.architecture) - 1:
        varMatrices["act_%s" %layer] = self.get_val(self.act[layer])

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
