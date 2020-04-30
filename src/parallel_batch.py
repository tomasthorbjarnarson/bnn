from milp.cplex_nn import get_cplex_nn
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.sat_margin import SAT_MARGIN
from milp.max_m import MAX_M
from gd.gd_nn import GD_NN
from helper.misc import infer_and_accuracy, clear_print, get_mean_bound_matrix, get_mean_vars,printProgressBar
from helper.data import load_data, get_batches, get_architecture
import argparse
import numpy as np
import time
from multiprocessing import Pool
import os
from pdb import set_trace

milps = {
  "min_w": MIN_W,
  "max_m": MAX_M,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "sat_margin": SAT_MARGIN
}

gds = {
  "gd_nn": GD_NN
}

solvers = {
  "gurobi": get_gurobi_nn,
  "cplex": get_cplex_nn
}

new_seeds = [8353134,14365767666,223454657,9734234,753283393493482349,473056832,3245823]

class Batch_Runner:
  def __init__(self, loss, data, batch_size, architecture, bound, reg, fair):
    self.loss = loss
    self.N = len(data["train_x"])
    self.batch_size = batch_size
    self.num_batches = self.N/batch_size
    self.architecture = architecture
    self.data = data
    self.bound = bound
    self.reg = reg
    self.fair = fair
    self.num_processes = len(os.sched_getaffinity(0))//2
    print("Number of processes to run on: %s." % self.num_processes)


  def run_batch(self, batch, bound_matrix, batch_num):
    nn = get_nn(milps[self.loss], batch, self.architecture, self.bound, self.reg, self.fair)
    nn.m.setParam('Threads', 2)
    nn.update_bounds(bound_matrix)
    nn.train(train_time*60, focus)
    runtime = nn.get_runtime()
    varMatrices = nn.extract_values()
    printProgressBar(batch_num+1, self.num_batches)

    return runtime, varMatrices

  def run_batches(self, batches, bound_matrix):
    batch_start = time.time()    

    pool = Pool(processes=self.num_processes)

    batch_nums = range(len(batches))
    bound_matrices = [bound_matrix for i in range(len(batches))]
    output = pool.starmap(self.run_batch, zip(batches, bound_matrices, batch_nums), chunksize=1)
    pool.close()
    pool.join()
    print("")

    batch_end = time.time()
    batch_time = batch_end - batch_start
    print("Time to run batches: %.3f" % (batch_time))
    #runtimes = [x[0] for x in output]
    network_vars = [x[1] for x in output]
    std_vars, total_std = get_std(network_vars)
    print("Total std: %.2f" % total_std)

    return network_vars

def get_std(network_vars):
  std_vars = {}
  std_sum = 0
  for key in network_vars[0]:
    if 'w_' in key or 'b_' in key:
      tmp_vars = np.stack([tmp[key] for tmp in network_vars])
      std_vars[key] = np.std(tmp_vars, axis=0)
      std_sum += std_vars[key].sum()
  return std_vars, std_sum

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--solver', default="gurobi", type=str)
  parser.add_argument('--hls', default='16', type=str)
  parser.add_argument('--ex', default=10, type=int)
  parser.add_argument('--focus', default=0, type=int)
  parser.add_argument('--time', default=1, type=float)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--loss', default="min_w", type=str)
  parser.add_argument('--data', default="mnist", type=str)
  parser.add_argument('--bound', default=1, type=int)
  parser.add_argument('--batch', default=0, type=int)
  parser.add_argument('--fair', default="", type=str)
  parser.add_argument('--reg', default=0, type=float)
  parser.add_argument('--epochs', default=5, type=int)
  args = parser.parse_args()
  
  solver = args.solver
  hls = [int(x) for x in args.hls.split("-") if len(args.hls) > 0]

  N = args.ex
  focus = args.focus
  train_time = args.time
  seed = args.seed
  loss = args.loss
  data_name = args.data
  bound = args.bound
  batch_size = args.batch
  reg = args.reg
  fair = args.fair
  epochs = args.epochs

  print(args)

  if loss not in milps and loss not in gds:
    raise Exception("MILP model %s not known" % loss)

  if solver not in solvers:
    raise Exception("Solver %s not known" % solver)

  data = load_data(data_name, N, seed)

  if batch_size == 0:
    batch_size = N
  batches = get_batches(data, batch_size)

  architecture = get_architecture(data, hls)
  print_str = "Architecture: %s. N: %s. Solver: %s. Loss: %s. Bound: %s"
  clear_print(print_str % ("-".join([str(x) for x in architecture]), N, solver, loss, bound))

  get_nn = solvers[solver]

  epoch_start = time.time()
  do_shuffle = True
  bound_matrix = {}

  batch_runner = Batch_Runner(loss, data, batch_size, architecture, bound, reg, fair)

  for epoch in range(epochs):
    if do_shuffle and epoch > 0:
      data = load_data(data_name, N, new_seeds[epoch-1])
      batches = get_batches(data, batch_size)

    clear_print("EPOCH %s" % epoch)
    printProgressBar(0, N/batch_size)
    network_vars = batch_runner.run_batches(batches, bound_matrix)

    bound_matrix = get_mean_bound_matrix(network_vars, bound, bound - epoch)
    mean_vars = get_mean_vars(network_vars)

    mean_train_acc = infer_and_accuracy(data['train_x'], data["train_y"], mean_vars, architecture)
    mean_val_acc = infer_and_accuracy(data['val_x'], data["val_y"], mean_vars, architecture)

    clear_print("Training accuracy for mean parameters: %s" % (mean_train_acc))
    clear_print("Validation accuracy for mean parameters: %s" % (mean_val_acc))


  total_time = time.time() - epoch_start
  print("Time to run all epochs: %.3f" % (total_time))

  final_train_acc = infer_and_accuracy(data['train_x'], data["train_y"], mean_vars, architecture)
  final_test_acc = infer_and_accuracy(data['test_x'], data["test_y"], mean_vars, architecture)

  clear_print("Training accuracy for mean parameters: %s" % (final_train_acc))
  clear_print("Testing accuracy for mean parameters: %s" % (final_test_acc))
  