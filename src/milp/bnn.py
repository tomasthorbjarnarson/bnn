"""
This work heavily relies on the implementation by Icarte:
https://bitbucket.org/RToroIcarte/bnn/src/master/
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from helper.mnist import load_mnist

def mycallback(model, where):
  if where == GRB.Callback.MIP:
    nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
    objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
    objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
    runtime = model.cbGet(GRB.Callback.RUNTIME)
    gap = 1 - objbnd/objbst
    if objbst < model._lastobjbst or objbnd > model._lastobjbnd:
      model._lastobjbst = objbst
      model._lastobjbnd = objbnd
      model._periodic.append((nodecnt, objbst, objbnd, runtime, gap))

class BNN:
  def __init__(self, N, architecture):

    self.N = N
    self.architecture = architecture
    self.train_x, self.train_y, self.oh_train_y, self.test_x, self.test_y, self.oh_test_y = load_mnist(N)
    dead = np.all(self.train_x == 0, axis=1)

    self.weights = {}
    self.biases = {}
    self.var_c = {}
    self.act = {}

    self.abs_weights = {}
    self.abs_biases = {}

    p_precision = GRB.INTEGER
    p_bound = 1

    pre_precision = GRB.CONTINUOUS
    pre_bound = 1

    act_precision = GRB.BINARY

    epsilon = 1

    self.m = gp.Model("mip1")

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
            self.weights[layer][i,j] = self.m.addVar(vtype=p_precision, lb=-p_bound, ub=p_bound, name="w_%s-%s_%s" % (layer,i,j))
            self.abs_weights[layer][i,j] = self.m.addVar(vtype=GRB.BINARY, name="abs(w)_%s-%s_%s" % (layer,i,j))
            self.m.addConstr(self.abs_weights[layer][i,j] >= self.weights[layer][i,j])
            self.m.addConstr(-self.abs_weights[layer][i,j] <= self.weights[layer][i,j])
          if layer > 1:
            # Var c only needed after first activation
            for k in range(N):
              self.var_c[layer][k,i,j] = self.m.addVar(vtype=pre_precision, lb=-pre_bound, ub=pre_bound, name="c_%s-%s_%s_%s" % (layer,i,j,k))
        # Bias only for each output neuron
        self.biases[layer][j] = self.m.addVar(vtype=p_precision, lb=-p_bound, ub=p_bound, name="b_%s-%s" % (layer,j))
        self.abs_biases[layer][j] = self.m.addVar(vtype=GRB.BINARY, name="abs(b)_%s-%s" % (layer,j))
        self.m.addConstr(self.abs_biases[layer][j] >= self.biases[layer][j])
        self.m.addConstr(-self.abs_biases[layer][j] <= self.biases[layer][j])

        if layer < len(architecture) - 1:
          for k in range(N):
            self.act[layer][k,j] = self.m.addVar(vtype=act_precision, name="act_%s-%s_%s" % (layer,j,k))


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
              self.m.addConstr(self.var_c[layer][k,i,j] - self.weights[layer][i,j] + 2*self.act[lastLayer][k,i] <= 2)
              self.m.addConstr(self.var_c[layer][k,i,j] + self.weights[layer][i,j] - 2*self.act[lastLayer][k,i] <= 0)
              self.m.addConstr(self.var_c[layer][k,i,j] - self.weights[layer][i,j] - 2*self.act[lastLayer][k,i] >= -2)
              self.m.addConstr(self.var_c[layer][k,i,j] + self.weights[layer][i,j] + 2*self.act[lastLayer][k,i] >= 0)
              inputs.append(self.var_c[layer][k,i,j])
          pre_activation = sum(inputs) + self.biases[layer][j]

          if layer < len(architecture) - 1:
            self.m.addConstr((self.act[layer][k,j] == 1) >> (pre_activation >= 0))
            self.m.addConstr((self.act[layer][k,j] == 0) >> (pre_activation <= -epsilon))
          else:
            if self.oh_train_y[j,k] > 0:
              self.m.addConstr(pre_activation >= 0)
            else:
              self.m.addConstr(pre_activation <= -epsilon)

    self.objSum = 0
    for layer in self.abs_weights:
      self.objSum += self.abs_weights[layer].sum()
    for layer in self.abs_biases:
      self.objSum += self.abs_biases[layer].sum()


  def train(self, time, focus):
    self.m.setObjective(self.objSum , GRB.MINIMIZE)
    self.m.setParam('TimeLimit', time)
    self.m.setParam('MIPFocus', focus)
    self.m._lastobjbst = GRB.INFINITY
    self.m._lastobjbnd = -GRB.INFINITY
    self.m._periodic = []
    self.m.optimize(mycallback)


  def extract_values(self):
    def getVal(maybeVar):
      try:
        val = maybeVar.x
      except:
        val = 0
      return val

    getVal = np.vectorize(getVal)

    varMatrices = {}
    for layer in self.weights:
      varMatrices["w_%s" %layer] = getVal(self.weights[layer])
      varMatrices["b_%s" %layer] = getVal(self.biases[layer])
      if layer > 1:
        varMatrices["c_%s" %layer] = getVal(self.var_c[layer])
      if layer < len(self.architecture) - 1:
        varMatrices["act_%s" %layer] = getVal(self.act[layer])

    return varMatrices


  def print_values(self):
    def getVal(maybeVar):
      try:
        val = maybeVar.x
      except:
        val = 0
      return val

    getVal = np.vectorize(getVal)

    for layer in self.weights:
      print("Weight %s" % layer)
      print(getVal(self.weights[layer]))
      print("Biases %s" % layer)
      print(getVal(self.biases[layer]))
      if layer > 1:
        print("C %s" % layer)
        print(getVal(self.var_c[layer]))
      if layer < len(self.architecture) - 1:
        print("Activations %s" % layer)
        print(getVal(self.act[layer]))
