"""
This work heavily relies on the implementation by Icarte:
https://bitbucket.org/RToroIcarte/bnn/src/master/
"""

import numpy as np
from helper.mnist import load_mnist
from globals import INT, BIN, CONT, EPSILON

class BNN:
  def __init__(self, model, N, architecture, obj, seed=0):

    self.N = N
    self.architecture = architecture
    self.obj = obj
    self.train_x, self.train_y, self.oh_train_y, self.val_x, self.val_y, self.oh_val_y, self.test_x, self.test_y, self.oh_test_y = load_mnist(N, seed)
    dead = np.all(self.train_x == 0, axis=1)
    self.sum_dead = np.sum(dead)

    self.weights = {}
    self.biases = {}
    self.var_c = {}
    self.act = {}

    self.abs_weights = {}
    self.abs_biases = {}


    bound = 1

    self.m = model

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
            self.weights[layer][i,j] = self.add_var(INT,"w_%s-%s_%s" % (layer,i,j), bound)
            self.abs_weights[layer][i,j] = self.add_var(BIN,"abs(w)_%s-%s_%s" % (layer,i,j))
            self.add_constraint(self.abs_weights[layer][i,j] >= self.weights[layer][i,j])
            self.add_constraint(-self.abs_weights[layer][i,j] <= self.weights[layer][i,j])
          if layer > 1:
            # Var c only needed after first activation
            for k in range(N):
              self.var_c[layer][k,i,j] = self.add_var(CONT,"c_%s-%s_%s_%s" % (layer,i,j,k), bound)
        # Bias only for each output neuron
        self.biases[layer][j] = self.add_var(INT,"b_%s-%s" % (layer,j), bound)
        self.abs_biases[layer][j] = self.add_var(BIN,"abs(b)_%s-%s" % (layer,j))
        self.add_constraint(self.abs_biases[layer][j] >= self.biases[layer][j])
        self.add_constraint(-self.abs_biases[layer][j] <= self.biases[layer][j])

        if layer < len(architecture) - 1:
          for k in range(N):
            self.act[layer][k,j] = self.add_var(BIN, name="act_%s-%s_%s" % (layer,j,k))

    if self.obj == "max_acc":
      self.output = np.full((N, architecture[-1]), None)
      for k in range(N):
        for j in range(architecture[-1]):
          self.output[k,j] = self.add_var(BIN, name="output_%s-%s" % (j,k))
        self.add_constraint(self.output[k,:].sum() == 1)

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
            self.add_constraint((self.act[layer][k,j] == 0) >> (pre_activation <= -EPSILON))
          elif self.obj == "max_acc":
            self.add_max_acc_output_constraints(j, k, pre_activation)
          else:
            self.add_output_constraints(j, k, pre_activation)

    if self.obj == "max_acc":
      self.calc_max_acc_objective()
    else:
      self.calc_objective()

  def add_max_acc_output_constraints(self, n_out, example, pre_activation):
    self.add_constraint((self.output[example,n_out] == 1) >> (pre_activation >= 0))
    self.add_constraint((self.output[example,n_out] == 0) >> (pre_activation <= -EPSILON))

  def add_output_constraints(self, n_out, example, pre_activation):
    if self.oh_train_y[n_out,example] > 0:
      self.add_constraint(pre_activation >= 0)
    else:
      self.add_constraint(pre_activation <= -EPSILON)

  def calc_objective(self):
    self.objSum = 0
    for layer in self.abs_weights:
      self.objSum += self.abs_weights[layer].sum()
    for layer in self.abs_biases:
      self.objSum += self.abs_biases[layer].sum()

  def calc_max_acc_objective(self):
    self.objSum = self.N
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        if self.oh_train_y[j,k] > 0:
          self.objSum -= self.output[k,j]
    #self.add_constraint(self.objSum >= np.floor(self.N*0.05))
    #self.regularizer = 0
    #for layer in self.abs_weights:
    #  self.regularizer += self.abs_weights[layer].sum()
    #for layer in self.abs_biases:
    #  self.regularizer += self.abs_biases[layer].sum()
    #from pdb import set_trace
    #set_trace()
    #self.add_constraint(self.regularizer <= (self.architecture[0]-self.sum_dead)*2.5)

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

    if self.obj == "max_acc":
      varMatrices["output"] = get_val(self.output)

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
