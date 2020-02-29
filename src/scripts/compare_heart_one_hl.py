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

time = 2*60
seeds = [959323,23421,46262544]
hl_neurons = 16
bound = 1
focus = 1

milps = {
  "min_w": MIN_W,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "min_hinge_reg": MIN_HINGE_REG
}

short = False
if short:
  num_examples = num_examples[0:2]
  seeds = [4,8]
  time = 2

def compare_heart_one_hl(losses, plot=False):

  all_results = {}
  num_experiments = len(losses)

  clear_print("Starting script, max time left: %s minutes" % get_time_left(1,1, num_experiments))

  i = 0
  json_dir = "results/json/compare_heart_one_hl"
  pathlib.Path(json_dir).mkdir(exist_ok=True)

  for loss in losses:
    i += 1
    print("Running %s nn experiments!" % loss)
    file_name = "%s-Time:%s-HL_Neurons:%s-Bound:%s-S:%s" % (loss, time, hl_neurons, bound, len(seeds))
    json_path = "%s/%s.json" % (json_dir, file_name)
    if pathlib.Path(json_path).is_file():
      print("Path %s exists" % json_path)
      with open(json_path, "r") as f:
        data = json.loads(f.read())
        all_results[loss] = data["results"]
      continue

    if loss in milps:
      all_results[loss] = run_experiments(loss, i, num_experiments)
    else:
      print("Loss %s unknown" % loss)
      continue
    with open(json_path, "w") as f:
      data = {"results": all_results[loss], "ts": datetime.now().strftime("%d-%m-%H:%M")}
      json.dump(data, f)

  x = num_examples
  plt.figure(1)
  for loss in losses:
    y, err = get_acc_mean_std(all_results[loss])
    plt.errorbar(x, y, yerr=err, capsize=3, label="%s test performance" % loss)
  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel("Test performance %")
  plt.title("Compare test accuracies for Heart Dataset with bound %s" % bound)
  plot_dir = "results/plots/compare_heart_one_hl/performance"
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
  if plot:
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  plt.figure(2)
  for loss in losses:
    y, err = get_runtime_mean_std(all_results[loss])
    plt.errorbar(x, y, yerr=err, capsize=3, label="%s runtime" % loss)
  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel("Runtime [s]")
  plt.title("Compare runtimes for Heart Dataset with bound %s" % bound)
  plot_dir = "results/plots/compare_heart_one_hl/runtimes"
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  if plot:
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

def run_experiments(loss, experiment, num_experiments):
  nn_results = []
  i = 0
  for N in num_examples:
    i += 1
    accs = []
    runtimes = []
    clear_print("Max time left: %s" % get_time_left(i, experiment, num_experiments))
    for s in seeds:
      data = load_data("heart", N, s)
      arch = get_architecture(data, [hl_neurons])

      clear_print("%s:  HL_Neurons: %s, N: %s, Seed: %s" % (loss, hl_neurons, N, s))
      nn = get_gurobi_nn(milps[loss], data, arch, bound)
      nn.train(60*time, focus)
      obj = nn.get_objective()
      runtime = nn.get_runtime()
      varMatrices = nn.extract_values()

      infer_test = inference(nn.data["test_x"], varMatrices, nn.architecture)
      test_acc = calc_accuracy(infer_test, nn.data["test_y"])
      accs.append(test_acc)
      runtimes.append(runtime)

    nn_results.append((accs, runtimes))

  return nn_results


def get_acc_mean_std(results):
  mean = [np.mean(z[0]) for z in results]
  std = [np.std(z[0]) for z in results]
  return mean, std

def get_runtime_mean_std(results):
  mean = [np.mean(z[1]) for z in results]
  std = [np.std(z[1]) for z in results]
  return mean, std

def get_time_left(example, experiment, num_experiments):
  time_left = len(num_examples[example-1:])*time*len(seeds)
  time_left += len(num_examples)*time*len(seeds)*(num_experiments - experiment)

  days = time_left // (60*24)
  time_left -= days*60*24
  hours = time_left // 60
  time_left -= hours*60
  minutes = time_left % 60

  return "%s days, %s hours, %s minutes" % (days, hours, minutes)
