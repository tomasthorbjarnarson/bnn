import numpy as np
from milp.nn import NN
from globals import EPSILON, BIN, TARGET_ERROR

class MAX_CORRECT(NN):
  def __init__(self, model, data, architecture, bound, reg):

    NN.__init__(self, model, data, architecture, bound, reg)

    self.out_bound = (self.architecture[-2]+1)*self.bound
    self.init_output()
    self.add_output_constraints()
    self.calc_objective()
    self.cutoff = np.prod(self.output.shape)*TARGET_ERROR

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
            inputs.append(self.train_x[k,i]*self.weights[layer][i,j])
          else:
            inputs.append(self.var_c[layer][k,i,j])
        pre_activation = sum(inputs) + self.biases[layer][j]
        pre_activation = 2*pre_activation/self.out_bound
        self.add_constraint((self.output[k,j] == 1) >> (pre_activation >= 0))
        self.add_constraint((self.output[k,j] == 0) >> (pre_activation <= -EPSILON))

  def calc_objective(self):
    self.obj = self.N
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        if self.oh_train_y[k,j] > 0:
          self.obj -= self.output[k,j]

    self.set_objective()

  def extract_values(self, get_func=lambda z: z.x):
    varMatrices = NN.extract_values(self, get_func)
    varMatrices["output"] = self.get_val(self.output, get_func)

    if self.reg:
      for layer in self.H:
        varMatrices["H_%s" % layer] = self.get_val(self.H[layer], get_func)

    return varMatrices
