import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
from datetime import datetime
from helper.misc import inference, calc_accuracy, clear_print, get_bound_matrix
from helper.data import load_data, get_batches, get_architecture
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.sat_margin import SAT_MARGIN

num_examples = [50, 100, 150, 200]

time = 5*60
seeds = [13, 37, 1111]
hl_neurons = 16
batch_size = 25
bound = 1
focus = 1

milps = {
  "min_w": MIN_W,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "sat_margin": SAT_MARGIN
}

max_data = False
if max_data:
  num_examples = [200]
  time = 5*60

short = False
if short:
  num_examples = [40, 60, 80, 100]
  seeds = [10,20]
  time = 10
  batch_size = 20

def compare_batch_training(losses, plot=False):
  if len(losses) > 1:
    raise Exception("This experiment is meant for only one loss")
  loss = losses[0]

  all_results = {}

  clear_print("Starting script, max time left: %s minutes" % get_time_left(0,0))

  json_dir = "results/json/compare_batch_training"
  pathlib.Path(json_dir).mkdir(exist_ok=True)
  file_name = "%s-Time:%s-HL_Neurons:%s-Loss:%s-S:%s-Prec:%s" % (loss, time, hl_neurons, loss, len(seeds), bound)
  json_path = "%s/%s.json" % (json_dir, file_name)
  if pathlib.Path(json_path).is_file():
    print("Path %s exists" % json_path)
    with open(json_path, "r") as f:
      data = json.loads(f.read())
      all_results = data["results"]
  else:
    print("Running %s nn experiments for all data!" % (loss))
    all_results["all"] = run_experiments(loss, False)
    print("Running %s nn experiments for all data!" % (loss))
    all_results["batch"] = run_experiments(loss, True)

    with open(json_path, "w") as f:
      data = {"results": all_results, "ts": datetime.now().strftime("%d-%m-%H:%M")}
      json.dump(data, f)

  x = num_examples
  plt.figure(1)

  y, err = get_acc_mean_std(all_results["all"])
  plt.errorbar(x, y, yerr=err, capsize=3, label="All data at once")
  y_batch, err_batch = get_acc_mean_std(all_results["batch"])
  plt.errorbar(x, y_batch, yerr=err_batch, capsize=3, label="Batch training")

  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel("Test performance %")
  plt.title("Test accuracies for loss %s, precision %s on Heart Dataset for all at once vs batch training" % (loss,bound))
  plot_dir = "results/plots/compare_batch_training/performance"
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
  if plot:
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  plt.figure(2)
  y, err = get_runtime_mean_std(all_results["all"])
  plt.errorbar(x, y, yerr=err, capsize=3, label="All data at once")
  y_batch, err_batch = get_runtime_mean_std(all_results["batch"])
  plt.errorbar(x, y_batch, yerr=err_batch, capsize=3, label="Batch training")
  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel("Runtime [s]")
  plt.title("Runtimes for loss %s, precision %s on Heart Dataset for all at once vs batch training" % (loss,bound))
  plot_dir = "results/plots/compare_batch_training/runtimes"
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  if plot:
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')


def run_experiments(loss, batch_train):
  nn_results = []
  i = 0
  for N in num_examples:
    accs = []
    runtimes = []
    clear_print("Max time left: %s" % get_time_left(i, batch_train))
    for s in seeds:
      data = load_data("heart", N, s)
      batches = get_batches(data, batch_size)
      arch = get_architecture(data, [hl_neurons])
      clear_print("%s:  HL_Neurons: %s, N: %s, Seed: %s, Batch: %s" % (loss, hl_neurons, N, s, batch_train))
      runtime = 0
      if batch_train:
        net_vars = []
        for batch in batches:
          batch_nn = get_gurobi_nn(milps[loss], batch, arch, bound)
          batch_nn.train(60*time, focus)
          runtime += batch_nn.get_runtime()
          varMatrices = batch_nn.extract_values()
          net_vars.append(varMatrices)
        bound_matrix = get_bound_matrix(net_vars, bound)
        nn = get_gurobi_nn(milps[loss], data, arch, bound)
        nn.update_bounds(bound_matrix)
        nn.train(60*time, focus)
      else:
        nn = get_gurobi_nn(milps[loss], data, arch, bound)
        nn.train(60*time, focus)

      obj = nn.get_objective()
      runtime += nn.get_runtime()
      varMatrices = nn.extract_values()

      infer_test = inference(nn.data["test_x"], varMatrices, nn.architecture)
      test_acc = calc_accuracy(infer_test, nn.data["test_y"])
      accs.append(test_acc)
      runtimes.append(runtime)

    nn_results.append((accs, runtimes))
    i += 1

  return nn_results


def get_acc_mean_std(results):
  mean = [np.mean(z[0]) for z in results]
  std = [np.std(z[0]) for z in results]
  return mean, std

def get_runtime_mean_std(results):
  mean = [np.mean(z[1]) for z in results]
  std = [np.std(z[1]) for z in results]
  return mean, std

def get_time_left(example, batch_train):
  time_left = len(num_examples[example:])*time*len(seeds)
  if not batch_train:
    time_left += len(num_examples)*time*len(seeds)

  days = time_left // (60*24)
  time_left -= days*60*24
  hours = time_left // 60
  time_left -= hours*60
  minutes = time_left % 60

  return "%s days, %s hours, %s minutes" % (days, hours, minutes)
