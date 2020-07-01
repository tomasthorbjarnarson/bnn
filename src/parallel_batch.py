from milp.cplex_nn import get_cplex_nn
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.sat_margin import SAT_MARGIN
from milp.max_m import MAX_M
from gd.gd_nn import GD_NN
from helper.misc import infer_and_accuracy, clear_print, get_weighted_mean_bound_matrix, get_mean_vars
from helper.misc import strip_network,printProgressBar, extract_params, get_weighted_mean_vars
from helper.data import load_data, get_training_batches, get_architecture
import argparse
import numpy as np
import time
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

focus=1
train_time=15
solver = "gurobi"


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

get_nn = solvers[solver]

new_seeds = [8353134,14365767666,223454657,9734234,753283393493482349,473056832,3245823,
             3842134,132414364572435,798452456,2132413245,788794342,134457678,213414,69797949393,
             34131413,46658765,1341324]

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

    del nn.m

    return runtime, extract_params(varMatrices, get_keys=["w","b","H"])[0]

  def run_batches(self, batches, bound_matrix):
    batch_start = time.time()    

    #pool = Pool(processes=self.num_processes)

    batch_nums = range(len(batches))
    bound_matrices = [bound_matrix for i in range(len(batches))]
    with Pool(processes=self.num_processes) as pool:
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

  def run_batch_alt(self, batch_data):
    batch = batch_data[0]
    bound_matrix = batch_data[1]
    batch_num = batch_data[2]
    nn = get_nn(milps[self.loss], batch, self.architecture, self.bound, self.reg, self.fair, batch=True)
    nn.m.setParam('Threads', 2)
    nn.update_bounds(bound_matrix)
    nn.train(train_time*60, focus)
    runtime = nn.get_runtime()
    varMatrices = nn.extract_values()
    dead = np.all(batch['train_x'] == 0, axis=0)
    printProgressBar(batch_num+1, self.num_batches)

    del nn.m
    del nn

    return runtime, extract_params(varMatrices, get_keys=["w","b","H"])[0], dead

  def run_batches_alt(self, batches, bound_matrix):
    batch_start = time.time()    

    pool = Pool(processes=self.num_processes)
    #batch_nums = range(len(batches))
    #bound_matrices = [bound_matrix for i in range(len(batches))]
    batch_data = []
    for i,batch in enumerate(batches):
      batch_data.append([batch, bound_matrix, i])
    output = pool.imap_unordered(self.run_batch_alt, batch_data)
    pool.close()
    pool.join()
    print("")

    batch_end = time.time()
    batch_time = batch_end - batch_start
    print("Time to run batches: %.3f" % (batch_time))
    #runtimes = [x[0] for x in output]
    all_results = [x for x in output]
    deads = [x[2] for x in all_results]
    network_vars = [x[1] for x in all_results]

    return network_vars, np.array(deads)

def get_std(network_vars):
  std_vars = {}
  std_sum = 0
  for key in network_vars[0]:
    if 'w_' in key or 'b_' in key:
      tmp_vars = np.stack([tmp[key] for tmp in network_vars])
      std_vars[key] = np.std(tmp_vars, axis=0)
      std_sum += std_vars[key].sum()
  return std_vars, std_sum

def run_parallel(seed, N, loss, data_name, batch_size, hls, bound, reg, fair, epochs):
  data = load_data(data_name, N, seed)

  if batch_size == 0:
    batch_size = N
  batches = get_training_batches(data, batch_size)

  architecture = get_architecture(data, hls)
  print_str = "Architecture: %s. N: %s. Solver: %s. Loss: %s. Bound: %s. Seed: %s."
  clear_print(print_str % ("-".join([str(x) for x in architecture]), N, solver, loss, bound, seed))


  epoch_start = time.time()
  do_shuffle = True
  bound_matrix = {}

  batch_runner = Batch_Runner(loss, data, batch_size, architecture, bound, reg, fair)
  all_avgs = []
  all_train_accs = []
  all_val_accs = []
  all_stds = []

  for epoch in range(epochs):
    if do_shuffle and epoch > 0:
      data = load_data(data_name, N, seed+new_seeds[epoch-1])
      batches = get_training_batches(data, batch_size)

    clear_print("EPOCH %s" % epoch)
    printProgressBar(0, N/batch_size)
    network_vars, deads = batch_runner.run_batches_alt(batches, bound_matrix)

    std_vars, total_std = get_std(network_vars)
    print("Total std: %.2f" % total_std)
    all_stds.append(total_std)
    
    if 'H_1' in network_vars[0]:
      neurons = [sum(x['H_1']) for x in network_vars]
      print("neurons", neurons)
      stripped_network_vars = []
      for var in network_vars:
        stripped_network, stripped_architecture = strip_network(var, architecture)
        stripped_network_vars.append(stripped_network)
      network_vars = stripped_network_vars
      batch_runner.architecture = stripped_architecture
      batch_runner.reg = 0

    val_accs = []
    for var in network_vars:
      val_acc = infer_and_accuracy(data["val_x"], data["val_y"], var, architecture)
      val_accs.append(val_acc)
    weighted_avg = get_weighted_mean_vars(network_vars, val_accs, deads)

    bound_matrix = get_weighted_mean_bound_matrix(network_vars, bound, bound - epoch, weighted_avg)

    mean_train_acc = infer_and_accuracy(data["train_x"], data["train_y"], weighted_avg, architecture)
    mean_val_acc = infer_and_accuracy(data["val_x"], data["val_y"], weighted_avg, architecture)

    all_avgs.append(weighted_avg)
    all_train_accs.append(mean_train_acc)
    all_val_accs.append(mean_val_acc)

    clear_print("Training accuracy for mean parameters: %s" % (mean_train_acc))
    clear_print("Validation accuracy for mean parameters: %s" % (mean_val_acc))


  total_time = time.time() - epoch_start
  print("Time to run all epochs: %.3f" % (total_time))

  best_ind = np.argmax(all_val_accs)
  best_avg = all_avgs[best_ind]

  final_train_acc = infer_and_accuracy(data["train_x"], data["train_y"], best_avg, architecture)
  final_val_acc = infer_and_accuracy(data["val_x"], data["val_y"], best_avg, architecture)
  final_test_acc = infer_and_accuracy(data["test_x"], data["test_y"], best_avg, architecture)

  clear_print("Final Training accuracy for mean parameters: %s" % (final_train_acc))
  clear_print("Final Validation accuracy for mean parameters: %s" % (final_val_acc))
  clear_print("Final Testing accuracy for mean parameters: %s" % (final_test_acc))

  return all_train_accs, all_val_accs, all_stds, final_train_acc, final_test_acc, total_time


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

  seeds = [1348612,7864568,9434861,3618393,93218484358]

  seed_train_accs = np.empty((len(seeds),epochs))
  seed_val_accs = np.empty((len(seeds),epochs))
  stds = np.empty((len(seeds),epochs))

  final_train_accs = []
  final_test_accs = []
  total_times = []
  
  for i,seed in enumerate(seeds):
    output = run_parallel(seed, N, loss, data_name, batch_size, hls, bound, reg, fair, epochs)
    all_train_accs, all_val_accs, all_stds, final_train_acc, final_test_acc, total_time  = output
    seed_train_accs[i] = all_train_accs
    seed_val_accs[i] = all_val_accs
    stds[i] = all_stds
    final_train_accs.append(final_train_acc)
    final_test_accs.append(final_test_acc)
    total_times.append(total_time)
  
  colors = sns.color_palette("husl", 3)
  sns.set_style("darkgrid")

  plt.figure(1)
  x = range(epochs)
  y = np.mean(seed_train_accs,axis=0)
  err = np.std(seed_train_accs, axis=0)
  plt.plot(x, y, color=colors[0], label="Train accuracy")
  plt.fill_between(x, y - err, y + err, alpha=0.3, facecolor=colors[0])

  y1 = np.mean(seed_val_accs,axis=0)
  err1 = np.std(seed_val_accs, axis=0)
  plt.plot(x, y1, color=colors[1], label="Validation accuracy")
  plt.fill_between(x, y1 - err1, y1 + err1, alpha=0.3, facecolor=colors[1])
  plt.legend()
  plt.ylim(70,100)
  plt.ylabel("Accuracy %")
  plt.xlabel("Epochs")
  plt.xticks(range(epochs))
  plt.title("Batch training - N: %s. Batch Size: %s. Loss: %s" % (N, batch_size, loss))
  plt.savefig("results/plots/Learning Curve - %s_%s" % (loss,datetime.now().strftime("%d-%m-%H:%M")))

  plt.figure(2)
  x = range(epochs)
  y = np.mean(stds,axis=0)
  err = np.std(stds, axis=0)
  plt.plot(x, y, color=colors[0], label="Standard Deviation")
  plt.fill_between(x, y - err, y + err, alpha=0.3, facecolor=colors[0])
  plt.legend()
  plt.xlabel("Epochs")
  plt.xticks(range(epochs))
  plt.title("Batch training - N: %s. Batch Size: %s. Loss: %s" % (N, batch_size, loss))
  plt.savefig("results/plots/STD Curve - %s_%s" % (loss,datetime.now().strftime("%d-%m-%H:%M")))

  print(seed_train_accs)
  print(seed_val_accs)
  print(stds)
  print(final_train_accs)
  print(final_test_accs)
  print(total_times)

  clear_print("Train: %s +/- %s" % (np.mean(final_train_accs), np.std(final_train_accs)))
  clear_print("Test: %s +/- %s" % (np.mean(final_test_accs), np.std(final_test_accs)))
  clear_print("Time: %s +/- %s" % (np.mean(total_times), np.std(total_times)))