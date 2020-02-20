import numpy as np
from milp.nn import NN
from globals import CONT

class MIN_HINGE(NN):
  def __init__(self, model, data, architecture, bound):

    NN.__init__(self, model, data, architecture, bound)

    self.out_bound = (self.architecture[-2]+1)*self.bound
    self.init_output()
    self.add_output_constraints()
    self.calc_objective()

  def init_output(self):
    self.output = np.full((self.N, self.architecture[-1]), None)
    for k in range(self.N):
      for j in range(self.architecture[-1]):
        self.output[k,j] = self.add_var(CONT, bound=self.out_bound, name="output_%s-%s" % (j,k))

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
        # Approximately normalize to between -1 and 1
        pre_activation = 2*pre_activation/self.out_bound
        self.add_constraint(self.output[k,j] == pre_activation*self.oh_train_y[k,j])

  def calc_objective(self):
    def hinge(u):
      return np.square(np.maximum(0, (0.5 - u)))
    npts = 2*self.out_bound+1
    #lb = -self.out_bound
    #ub = self.out_bound
    lb = -1
    ub = 1
    ptu = []
    pthinge = []
    for i in range(npts):
      ptu.append(lb + (ub - lb) * i / (npts-1))
      pthinge.append(hinge(ptu[i]))

    for k in range(self.N):
      for j in range(self.architecture[-1]):
        self.m.setPWLObj(self.output[k,j], ptu, pthinge)

  def extract_values(self):
    varMatrices = NN.extract_values(self)
    varMatrices["output"] = self.get_val(self.output)

    return varMatrices
