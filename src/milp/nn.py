"""
This work heavily relies on the implementation by Icarte:
https://bitbucket.org/RToroIcarte/bnn/src/master/
"""

import numpy as np
from globals import INT, BIN, CONT, EPSILON

class NN:
  def __init__(self, model, data, architecture, bound, reg, fair, batch=False):

    self.N = len(data["train_x"])
    self.architecture = architecture
    self.data = data
    self.train_x = data["train_x"]
    self.oh_train_y = data["oh_train_y"]

    self.bound = bound
    self.reg = reg
    self.fair = fair
    self.batch = batch

    self.m = model

    if len(architecture) > 2:
      self.out_bound = (self.architecture[-2]+1)*self.bound
    else:
      self.out_bound = np.mean(data['train_x'])*architecture[0]

    self.init_params()
    self.add_examples()
    if fair in ["EO", "DP"]:
      self.add_fairness()

  def init_params(self):
    self.weights = {}
    self.biases = {}
    self.var_c = {}
    self.act = {}

    # All pixels that are 0 in every example are considered dead
    self.dead = np.all(self.train_x == 0, axis=0)
    if self.batch:
      self.dead = np.all(self.train_x == 10000, axis=0)

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
          if layer == 1 and self.dead[i]:
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
            self.act[layer][k,j] = self.add_var(BIN, "act_%s-%s_%s" % (layer,j,k))

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

  def add_regularizer(self):
    self.H = {}

    for lastLayer, neurons_out in enumerate(self.architecture[1:-1]):
      layer = lastLayer + 1
      neurons_in = self.architecture[lastLayer]

      self.H[layer] = np.full(neurons_out, None)
      for j in range(neurons_out):
        self.H[layer][j] = self.add_var(BIN, "h_%s-%s" % (layer,j))
        for i in range(neurons_in):
          if not (layer == 1 and self.dead[i]):
            self.add_constraint((self.H[layer][j] == 0) >> (self.weights[layer][i,j] == 0))
        self.add_constraint((self.H[layer][j] == 0) >> (self.biases[layer][j] == 0))
        for n in range(self.architecture[layer+1]):
          self.add_constraint((self.H[layer][j] == 0) >> (self.weights[layer+1][j,n] == 0))
          #for k in range(self.N):
            #self.add_constraint((self.H[layer][j] == 0) >> (self.var_c[layer+1][k,j,n] == 0))
            #self.add_constraint((self.H[layer][j] == 0) >> (self.act[layer][j] == 1))

    # Last hidden layer should have at least as many neurons as the output layer
    self.add_constraint(self.H[layer].sum() >= self.architecture[-1])

  def add_fairness(self):
    layer = len(self.architecture) - 1
    lastLayer = layer - 1
    neurons_in = self.architecture[lastLayer]
    neurons_out = self.architecture[layer]

    self.pred_labels = np.full(self.N, None)
    for k in range(self.N):
      self.pred_labels[k] = self.add_var(BIN, name="label_%s" % k)

    females = self.data['train_x'][:,64]
    males = self.data['train_x'][:,65]
    labels = self.data['train_y']
    false_labels = 1 - labels

    for k in range(self.N):
      pre_acts = 0
      for j in range(neurons_out):
        inputs = []
        for i in range(neurons_in):
          if layer == 1:
            inputs.append(self.train_x[k,i]*self.weights[layer][i,j])
          else:
            inputs.append(self.var_c[layer][k,i,j])
        pre_activation = sum(inputs) + self.biases[layer][j]
        pre_activation = 2*pre_activation/self.out_bound

        if j == 0:
          pre_acts += pre_activation
        else:
          pre_acts -= pre_activation

      self.add_constraint((self.pred_labels[k] == 0) >> (pre_acts >= 0))
      self.add_constraint((self.pred_labels[k] == 1) >> (pre_acts <= 0))

    if self.fair == "EO":
      self.female_pred1_true1 = (females*labels*self.pred_labels).sum() / (females*labels).sum() 
      self.male_pred1_true1 = (males*labels*self.pred_labels).sum() / (males*labels).sum()

      self.female_pred1_true0 = (females*false_labels*self.pred_labels).sum() / (females*false_labels).sum() 
      self.male_pred1_true0 = (males*false_labels*self.pred_labels).sum() / (males*false_labels).sum()

      fair_constraint = 0.02
      self.add_constraint(self.female_pred1_true1 - self.male_pred1_true1 <= fair_constraint)
      self.add_constraint(self.female_pred1_true1 - self.male_pred1_true1 >= -fair_constraint)
      self.add_constraint(self.female_pred1_true0 - self.male_pred1_true0 <= fair_constraint)
      self.add_constraint(self.female_pred1_true0 - self.male_pred1_true0 >= -fair_constraint)
    elif self.fair == "DP":
      self.female_pred1 = (females*self.pred_labels).sum() / females.sum() 
      self.male_pred1 = (males*self.pred_labels).sum() / males.sum()

      fair_constraint = 0.05
      #self.add_constraint(self.female_pred1 - self.male_pred1 <= fair_constraint)
      #self.add_constraint(self.female_pred1 - self.male_pred1 >= -fair_constraint)
      self.add_constraint(self.female_pred1 >= 0.8*self.male_pred1)
      self.add_constraint(self.male_pred1 >= 0.8*self.female_pred1)

  def update_bounds(self, bound_matrix={}):
    for lastLayer, neurons_out in enumerate(self.architecture[1:]):
      layer = lastLayer + 1
      neurons_in = self.architecture[lastLayer]

      for j in range(neurons_out):
        for i in range(neurons_in):
          if "w_%s_lb" % layer in bound_matrix and type(self.weights[layer][i,j]) != int:
            self.weights[layer][i,j].lb = bound_matrix["w_%s_lb" % layer][i,j]
          if "w_%s_ub" % layer in bound_matrix and type(self.weights[layer][i,j]) != int:
            self.weights[layer][i,j].ub = bound_matrix["w_%s_ub" % layer][i,j]

        if "b_%s_lb" % layer in bound_matrix:
          self.biases[layer][j].lb = bound_matrix["b_%s_lb" % layer][j]
        if "b_%s_ub" % layer in bound_matrix:
          self.biases[layer][j].ub = bound_matrix["b_%s_ub" % layer][j]

  def warm_start(self, varMatrices):
    # All pixels that are 0 in every example are considered dead
    dead = np.all(self.train_x == 0, axis=0)

    for lastLayer, neurons_out in enumerate(self.architecture[1:]):
      layer = lastLayer + 1
      neurons_in = self.architecture[lastLayer]

      for j in range(neurons_out):
        for i in range(neurons_in):
          if not (layer == 1 and dead[i]):
            self.weights[layer][i,j].start = varMatrices["w_%s" % layer][i,j]
        # Bias only for each output neuron
        self.biases[layer][j].start = varMatrices["b_%s" % layer][j]


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

  def extract_values(self, get_func=lambda z: z.x):
    varMatrices = {}
    for layer in self.weights:
      varMatrices["w_%s" %layer] = self.get_val(self.weights[layer], get_func)
      varMatrices["b_%s" %layer] = self.get_val(self.biases[layer], get_func)
      if layer > 1:
        varMatrices["c_%s" %layer] = self.get_val(self.var_c[layer], get_func)
      if layer < len(self.architecture) - 1:
        varMatrices["act_%s" %layer] = self.get_val(self.act[layer], get_func)

    if self.fair in ["EO", "DP"]:
      varMatrices["pred_labels"] = self.get_val(self.pred_labels, get_func)

    return varMatrices


  def print_values(self):
    for layer in self.weights:
      print("Weight %s" % layer)
      print(self.get_val(self.weights[layer]))
      print("Biases %s" % layer)
      print(self.get_val(self.biases[layer]))
      if layer > 1:
        print("C %s" % layer)
        print(self.get_val(self.var_c[layer]))
      if layer < len(self.architecture) - 1:
        print("Activations %s" % layer)
        print(self.get_val(self.act[layer]))
