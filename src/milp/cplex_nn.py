from docplex.mp.model import Model
from docplex.mp.progress import SolutionListener
import numpy as np
import math
from helper.misc import inference, calc_accuracy
from globals import INT, BIN, CONT, LOG

def get_cplex_nn(NN, data, architecture, bound):
  # Init a NN using CPLEX API according to the NN supplied
  class Cplex_NN(NN):
    def __init__(self, data, architecture, bound):
      self.log = LOG
      model = Model("Cplex_NN", log_output=LOG)
      NN.__init__(self, model, data, architecture, bound)
      
    def add_var(self, precision, name, bound=0):
      if precision == INT:
        return self.m.integer_var(lb=-bound, ub=bound, name=name)
      elif precision == BIN:
        return self.m.binary_var(name=name)
      elif precision == CONT:
        return self.m.continuous_var(lb=-bound, ub=bound, name=name)
      else:
        raise Exception('Parameter precision not known: %s' % precision)

    def add_constraint(self, constraint):
      self.m.add_constraint(constraint)

    def set_objective(self):
      pass

    def train(self, time=None, focus=None):
      if time:
        self.m.set_time_limit(time)
      if focus:
        self.m.context.cplex_parameters.emphasis.mip = focus
      if self.log:
        self.m.print_information()
      self.m.minimize(self.obj)
      listener = MyProgressListener(self.m, self.weights, self.biases, self.data["val_x"], self.data["val_y"], self.architecture)
      self.m.add_progress_listener(listener)
      self.sol = self.m.solve()
      if self.log:
        self.m.report()
      self.progress = listener.get_progress()
        
    def get_objective(self):
      return self.sol.objective_value

    def get_runtime(self):
      return self.m.get_solve_details().time

    def get_data(self):
      data = {
        'obj': self.get_objective(),
        'bound': self.m.solve_details.best_bound,
        'gap': self.m.solve_details.mip_relative_gap,
        'nodecount': self.m.solve_details.nb_nodes_processed,
        'num_vars': self.m.number_of_variables,
        'num_int_vars': self.m.number_of_integer_variables,
        'num_binary_vars': self.m.number_of_binary_variables,
        'num_constrs': self.m.number_of_constraints,
        'num_nonzeros': self.m.solve_details.nb_linear_nonzeros,
        'periodic': self.progress,
        'variables': self.extract_values(),
      }
      return data

    def get_val(self, maybe_var):
      try:
        val = self.sol.get_value(maybe_var)
      except:
        val = 0
      return val


  class MyProgressListener(SolutionListener):
    def __init__(self, model, weights, biases, val_x, val_y, architecture):
      SolutionListener.__init__(self, model)
      self.solutions = []
      self.progress = []
      self.lastobjbst = math.inf
      self.lastobjbnd = -math.inf
      self.last_gap = 1
      self.last_obj = math.inf
      self.val_acc = 0
      self.model = model
      self.weights = weights
      self.biases = biases
      self.val_x = val_x
      self.val_y = val_y
      self.architecture = architecture

    def notify_solution(self, s):
      SolutionListener.notify_solution(self, s)
      sol = self.current_solution
      self.solutions.append(sol)
      if sol.objective_value != self.last_obj:
        varMatrices = {}
        for layer in self.weights:
          w_shape = self.weights[layer].shape
          b_shape = self.biases[layer].shape
          varMatrices["w_%s" % layer] = np.zeros(w_shape)
          varMatrices["b_%s" % layer] = np.zeros(b_shape)
          for i in range(w_shape[0]):
            for j in range(w_shape[1]):
              if type(self.weights[layer][i,j]) != int:
                varMatrices["w_%s" % layer][i,j] = sol.get_value(self.weights[layer][i,j])
          for i in range(b_shape[0]):
            varMatrices["b_%s" % layer][i] = sol.get_value(self.biases[layer][i])

        infer_test = inference(self.val_x, varMatrices, self.architecture)
        val_acc = calc_accuracy(infer_test, self.val_y)
        #print("Validation accuracy: %s " % (val_acc))

        self.last_obj = sol.objective_value
        self.val_acc = val_acc

    def notify_progress(self, progress_data):
      SolutionListener.notify_progress(self, progress_data)
      nodecnt = progress_data.current_nb_nodes
      objbst  = progress_data.current_objective
      objbnd  = progress_data.best_bound
      runtime = progress_data.time
      gap     = progress_data.mip_gap
      if self.last_gap - gap > 0.01 and (objbst < self.lastobjbst or objbnd > self.lastobjbnd):
        self.progress.append((nodecnt, objbst, objbnd, runtime, gap, self.val_acc))
        self.last_gap = gap

    def notify_end(self, status, objective):
      if status:
        nodecnt = self.model.solve_details.nb_nodes_processed
        objbst  = objective
        objbnd  = objective
        runtime = self.model.solve_details.time
        gap     = 0
        self.progress.append((nodecnt, objbst, objbnd, runtime, gap, self.val_acc))

    def get_solutions(self):
      return self.solutions

    def get_progress(self):
      return self.progress

  return Cplex_NN(data, architecture, bound)
