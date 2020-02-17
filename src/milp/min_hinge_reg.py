import numpy as np
from milp.nn import NN
from globals import CONT, BIN


class MIN_HINGE_REG(NN):
  def __init__(self, model, data, architecture, bound):

    NN.__init__(self, model, data, architecture, bound)

    self.init_output()
    self.add_output_constraints()
    self.add_regularizer()
    self.calc_objective()

  def add_regularizer(self):
    self.H = {}
    dead = np.all(self.train_x == 0, axis=0)

    for lastLayer, neurons_out in enumerate(self.architecture[1:-1]):
      layer = lastLayer + 1
      neurons_in = self.architecture[lastLayer]

      self.H[layer] = np.full(neurons_out, None)
      for j in range(neurons_out):
        self.H[layer][j] = self.add_var(BIN, "h_%s-%s" % (layer,j))
        for i in range(neurons_in):
          if not (layer == 1 and dead[i]):
            self.add_constraint((self.H[layer][j] == 0) >> (self.weights[layer][i,j] == 0))
        self.add_constraint((self.H[layer][j] == 0) >> (self.biases[layer][j] == 0))
        for n in range(self.architecture[layer+1]):
          self.add_constraint((self.H[layer][j] == 0) >> (self.weights[layer+1][j,n] == 0))

  def init_output(self):
    self.out_bound = (self.architecture[-2]+1)*self.bound
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
        # Approximately normalize to between 0 and 1
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

    if len(self.architecture) > 2:
      alpha = 1/sum(self.architecture[1:-1])
      self.obj = 0
      for layer in self.H:
        self.obj += self.H[layer].sum()*alpha
      self.add_constraint(self.obj >= 10*alpha)

      self.set_objective()

    for k in range(self.N):
      for j in range(self.architecture[-1]):
        self.m.setPWLObj(self.output[k,j], ptu, pthinge)


  def extract_values(self):
    varMatrices = NN.extract_values(self)
    varMatrices["output"] = self.get_val(self.output)
    for layer in self.H:
      varMatrices["H_%s" % layer] = self.get_val(self.H[layer])

    return varMatrices
