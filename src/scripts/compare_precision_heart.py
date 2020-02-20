import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
from datetime import datetime
from helper.misc import inference, calc_accuracy, clear_print
from helper.data import load_data, get_architecture
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.min_hinge_reg import MIN_HINGE_REG

num_examples = [50, 100, 150, 200]
bounds = [1, 3, 7, 15]

time = 2*60
seeds = [1,2,3]
hl_neurons = 20

milps = {
  "min_w": MIN_W,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "min_hinge_reg": MIN_HINGE_REG
}

short = False
if short:
  num_examples = [40, 60, 80]
  bounds = [1,3,7]
  seeds = [10,20]
  time = 2

def compare_precision_heart(losses, plot=False):
  if len(losses) > 1:
    raise Exception("This experiment is meant for only one loss")
  loss = losses[0]

  all_results = {}

  clear_print("Starting script, max time left: %s minutes" % get_time_left(0,0))

  json_dir = "results/json/compare_precision_heart"
  pathlib.Path(json_dir).mkdir(exist_ok=True)
  file_name = "%s-Time:%s-HL_Neurons:%s-Loss:%s-S:%s-Prec:%s" % (loss, time, hl_neurons, loss, len(seeds), len(bounds))
  json_path = "%s/%s.json" % (json_dir, file_name)
  if pathlib.Path(json_path).is_file():
    print("Path %s exists" % json_path)
    with open(json_path, "r") as f:
      data = json.loads(f.read())
      all_results = data["results"]
  else:
    i = 0
    for bound in bounds:
      print("Running %s nn experiments for bound %s!" % (loss, bound))
      i += 1
      all_results[str(bound)] = run_experiments(loss, bound)

    with open(json_path, "w") as f:
      data = {"results": all_results, "ts": datetime.now().strftime("%d-%m-%H:%M")}
      json.dump(data, f)

  x = num_examples
  plt.figure(1)
  for bound in bounds:
    y, err = get_acc_mean_std(all_results[str(bound)])
    plt.errorbar(x, y, yerr=err, capsize=1, label="Test performance for precision: %s" % bound)
  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel("Test performance %")
  plt.title("Compare test accuracies for loss %s on Heart Dataset for different precisions" % loss)
  plot_dir = "results/plots/compare_precision_heart/performance"
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
  if plot:
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  plt.figure(2)
  for bound in bounds:
    y, err = get_runtime_mean_std(all_results[str(bound)])
    plt.errorbar(x, y, yerr=err, capsize=1, label="Runtime for precision: %s" % bound)
  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel("Runtime [s]")
  plt.title("Compare runtimes for loss %s on Heart Dataset for different precisions" % loss)
  plot_dir = "results/plots/compare_precision_heart/runtimes"
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  if plot:
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

def run_experiments(loss, bound):
  nn_results = []
  i = 0
  for N in num_examples:
    accs = []
    runtimes = []
    clear_print("Max time left: %s" % get_time_left(i, bounds.index(bound)))
    for s in seeds:
      data = load_data("heart", N, s)
      arch = get_architecture(data, [hl_neurons])
      in_neurons = [data["train_x"].shape[1]]
      out_neurons = [data["oh_train_y"].shape[1]]
      arch = in_neurons + [hl_neurons] + out_neurons

      clear_print("%s:  HL_Neurons: %s, N: %s, Seed: %s, Bound: %s" % (loss, hl_neurons, N, s, bound))
      nn = get_gurobi_nn(milps[loss], data, arch, bound)
      nn.train(60*time, 0)
      obj = nn.get_objective()
      runtime = nn.get_runtime()
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

def get_time_left(example, bound_index):
  time_left = len(num_examples[example:])*time*len(seeds)
  time_left += len(num_examples)*time*len(seeds)*(len(bounds) - bound_index - 1)

  days = time_left // (60*24)
  time_left -= days*60*24
  hours = time_left // 60
  time_left -= hours*60
  minutes = time_left % 60

  return "%s days, %s hours, %s minutes" % (days, hours, minutes)
