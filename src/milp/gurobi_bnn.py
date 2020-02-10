import gurobipy as gp
from gurobipy import GRB
import numpy as np
from helper.misc import inference, calc_accuracy
from globals import INT, BIN, CONT, LOG

def get_gurobi_bnn(BNN, data, architecture, bound):
  # Init a BNN using Gurobi API according to the BNN supplied
  class Gurobi_BNN(BNN):
    def __init__(self, data, architecture, bound):
      model = gp.Model("Gurobi_BNN")
      if not LOG:
        model.setParam("OutputFlag", 0)
      BNN.__init__(self, model, data, architecture, bound)
      
    def add_var(self, precision, name, bound=0):
      if precision == INT:
        return self.m.addVar(vtype=GRB.INTEGER, lb=-bound, ub=bound, name=name)
      elif precision == BIN:
        return self.m.addVar(vtype=GRB.BINARY, name=name)
      elif precision == CONT:
        return self.m.addVar(vtype=GRB.CONTINUOUS, lb=-bound, ub=bound, name=name)
      else:
        raise Exception('Parameter precision not known: %s' % precision)

    def add_constraint(self, constraint):
      self.m.addConstr(constraint)

    def set_objective(self):
      self.m.setObjective(self.obj, GRB.MINIMIZE)
      
    def train(self, time=None, focus=None):
      if time:
        self.m.setParam('TimeLimit', time)
      if focus:
        self.m.setParam('MIPFocus', focus)
      #self.m.setParam('Threads', 1)
      self.m._lastobjbst = GRB.INFINITY
      self.m._lastobjbnd = -GRB.INFINITY
      self.m._progress = []
      self.m._weights = self.weights
      self.m._biases = self.biases
      self.m._data = self.data
      self.m._architecture = self.architecture
      self.m._val_acc = 0
      self.m.update()
      self.m.optimize(mycallback)

      # Add last value after optimisation finishes
      gap = 1 - self.m.ObjBound/(self.m.ObjVal + 1e-15)
      if gap != self.m._progress[-1][4]:
        self.m._progress.append((self.m.NodeCount, self.m.ObjVal, self.m.ObjBound, self.m.Runtime, gap))

    def get_objective(self):
      return self.m.ObjVal

    def get_runtime(self):
      return self.m.Runtime

    def get_data(self):
      data = {
        'obj': self.m.ObjVal,
        'bound': self.m.ObjBound,
        'gap': self.m.MIPGap,
        'nodecount': self.m.NodeCount,
        'num_vars': self.m.NumVars,
        'num_int_vars': self.m.NumIntVars - self.m.NumBinVars,
        'num_binary_vars': self.m.NumBinVars,
        'num_constrs': self.m.NumConstrs,
        'num_nonzeros': self.m.NumNZs,
        'periodic': self.m._progress,
        'variables': self.extract_values(),
      }
      return data

    def get_val(self, maybe_var):
      tmp = np.zeros(maybe_var.shape)
      for index, count in np.ndenumerate(maybe_var):
        try:
          # Sometimes solvers have "integer" values like 1.000000019, round it to 1
          if maybe_var[index].VType == 'I':
            tmp[index] = round(maybe_var[index].x)
          else:
            tmp[index] = maybe_var[index].x
        except:
          tmp[index] = 0
      return tmp

  return Gurobi_BNN(data, architecture, bound)


def mycallback(model, where):
  if where == GRB.Callback.MIP:
    nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
    objbst = model.cbGet(GRB.Callback.MIP_OBJBST) + 1e-15
    objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
    runtime = model.cbGet(GRB.Callback.RUNTIME)
    gap = 1 - objbnd/objbst
    if objbst < model._lastobjbst or objbnd > model._lastobjbnd:
      model._lastobjbst = objbst
      model._lastobjbnd = objbnd
      model._progress.append((nodecnt, objbst, objbnd, runtime, gap, model._val_acc))
  elif where == GRB.Callback.MIPSOL:
    nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
    objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST) + 1e-15
    objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
    runtime = model.cbGet(GRB.Callback.RUNTIME)
    gap = 1 - objbnd/objbst
    model._lastobjbst = objbst
    model._lastobjbnd = objbnd

    varMatrices = {}
    for layer in model._weights:
      w_shape = model._weights[layer].shape
      b_shape = model._biases[layer].shape
      varMatrices["w_%s" % layer] = np.zeros(w_shape)
      varMatrices["b_%s" % layer] = np.zeros(b_shape)
      for i in range(w_shape[0]):
        for j in range(w_shape[1]):
          if type(model._weights[layer][i,j]) != int:
            varMatrices["w_%s" % layer][i,j] = model.cbGetSolution(model._weights[layer][i,j])
      for i in range(b_shape[0]):
        varMatrices["b_%s" % layer][i] = model.cbGetSolution(model._biases[layer][i])

    infer_train = inference(model._data['train_x'], varMatrices, model._architecture)
    infer_val = inference(model._data['val_x'], varMatrices, model._architecture)
    train_acc = calc_accuracy(infer_train, model._data['train_y'])
    val_acc = calc_accuracy(infer_val, model._data['val_y'])
    if LOG:
      print("Train accuracy: %s " % (train_acc))
      print("Validation accuracy: %s " % (val_acc))

    model._progress.append((nodecnt, objbst, objbnd, runtime, gap, val_acc))
    model._val_acc = val_acc

    # Maybe remove later
    if train_acc == 100 and objbst < 1:
      model.terminate()