import numpy as np
from milp.bnn import BNN
from globals import EPSILON, BIN

class MAX_CORRECT_BNN(BNN):
  def __init__(self, model, N, architecture, seed=0):

    BNN.__init__(self, model, N, architecture, seed)

    self.init_output()
    self.add_output_constraints()
    self.calc_objective()

  def init_output(self):
    self.output = np.full((self.N, self.architecture[-1]), None)
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        self.output[k,j] = self.add_var(BIN, name="output_%s-%s" % (j,k))
      self.add_constraint(self.output[k,:].sum() == 1)

  def add_output_constraints(self):
    layer = len(self.architecture) - 1
    lastLayer = layer - 1
    neurons_in = self.architecture[lastLayer]
    neurons_out = self.architecture[layer]

    for k in range(self.N):
      for j in range(neurons_out):
        inputs = []
        for i in range(neurons_in):
          if layer == 1:
            inputs.append(self.train_x[i,k]*self.weights[layer][i,j])
          else:
            inputs.append(self.var_c[layer][k,i,j])
        pre_activation = sum(inputs) + self.biases[layer][j]
        self.add_constraint((self.output[k,j] == 1) >> (pre_activation >= 0))
        self.add_constraint((self.output[k,j] == 0) >> (pre_activation <= -EPSILON))    

  def calc_objective(self):
    self.obj = self.N
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        if self.oh_train_y[j,k] > 0:
          self.obj -= self.output[k,j]

    self.set_objective()

  def extract_values(self):
    get_val = np.vectorize(self.get_val)
    varMatrices = BNN.extract_values(self)
    varMatrices["output"] = get_val(self.output)

    return varMatrices
