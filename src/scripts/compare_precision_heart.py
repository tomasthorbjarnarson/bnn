import matplotlib.pyplot as plt
import numpy as np
import pathlib
import seaborn as sns
import json
from datetime import datetime
from helper.misc import infer_and_accuracy, clear_print
from helper.data import load_data, get_architecture
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.sat_margin import SAT_MARGIN

num_examples = [40, 80, 120, 160, 200]
bounds = [1, 3, 7, 15]

time = 10*60
seeds = [11,34234,114341]
hl_neurons = 16
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

short = True
if short:
  num_examples = [40, 80, 120]
  bounds = [1,3,7]
  seeds = [1348612,7864568]
  time = 5

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

  ylabel = "Accuracy %"
  plot_title = "Train accuracies for loss %s on Heart Dataset for different precisions" % loss
  plot_dir = "results/plots/compare_precision_heart/%s"
  plot_results(all_results, "train_accs", bounds, 1, ylabel, plot_title, plot_dir % "train_acc", file_name)

  plot_title = "Test accuracies for loss %s on Heart Dataset for different precisions" % loss
  plot_results(all_results, "test_accs", bounds, 2, ylabel, plot_title, plot_dir % "test_acc", file_name)

  ylabel = "Runtime [s]"
  plot_title = "Runtimes for loss %s on Heart Dataset for different precisions" % loss
  plot_results(all_results, "runtimes", bounds, 3, ylabel, plot_title, plot_dir % "runtimes", file_name)

def run_experiments(loss, bound):
  nn_results = {
    "train_accs": {},
    "test_accs": {},
    "runtimes": {},
    "objs": {}
  }
  i = 0
  for N in num_examples:
    nn_results["train_accs"][N] = []
    nn_results["test_accs"][N] = []
    nn_results["runtimes"][N] = []
    nn_results["objs"][N] = []
    clear_print("Max time left: %s" % get_time_left(i, bounds.index(bound)))
    for s in seeds:
      data = load_data("heart", N, s)
      arch = get_architecture(data, [hl_neurons])
      in_neurons = [data["train_x"].shape[1]]
      out_neurons = [data["oh_train_y"].shape[1]]
      arch = in_neurons + [hl_neurons] + out_neurons

      clear_print("%s:  HL_Neurons: %s, N: %s, Seed: %s, Bound: %s" % (loss, hl_neurons, N, s, bound))
      nn = get_gurobi_nn(milps[loss], data, arch, bound)
      nn.train(60*time, focus)
      obj = nn.get_objective()
      runtime = nn.get_runtime()
      varMatrices = nn.extract_values()
      train_acc = infer_and_accuracy(nn.data["train_x"], nn.data["train_y"], varMatrices, nn.architecture)
      test_acc = infer_and_accuracy(nn.data["test_x"], nn.data["test_y"], varMatrices, nn.architecture)

      nn_results["train_accs"][N].append(train_acc)
      nn_results["test_accs"][N].append(test_acc)
      nn_results["runtimes"][N].append(runtime)
      nn_results["objs"][N].append(obj)

    i += 1

  return nn_results

def plot_results(all_results, result_type, bounds, index, ylabel, title, plot_dir, file_name):
  x = num_examples
  plt.figure(index)
  colors = sns.color_palette("husl", len(bounds))
  i = 0
  for bound in bounds:
    y, err = get_mean_std(all_results[str(bound)][result_type].values())
    #plt.errorbar(x, y, yerr=err, capsize=3, label="Precision: %s" % bound)
    plt.plot(x,y, label="Precision: %s" % bound, color = colors[i])
    plt.fill_between(x, y - err, y + err, alpha=0.3, facecolor=colors[i])
    i += 1
  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel(ylabel)
  plt.title(title)
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
  plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

def get_mean_std(results):
  mean = np.array([np.mean(z) for z in results])
  std = np.array([np.std(z) for z in results])
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
