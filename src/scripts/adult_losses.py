import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
import seaborn as sns
from datetime import datetime
from helper.misc import infer_and_accuracy, clear_print
from helper.data import load_data, get_architecture
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.max_margin import MAX_MARGIN

num_examples = [40, 80, 120, 160, 200, 240, 280]

time = 10*60
seeds = [1348612,7864568,9434861]
hl_neurons = 16
bound = 1
focus = 1

milps = {
  "min_w": MIN_W,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "max_margin": MAX_MARGIN
}

short = True
if short:
  num_examples = [20,40,60,80,100,120,140]
  seeds = [745,648]
  time = 10

def adult_losses(losses, plot=False):

  all_results = {}
  num_experiments = len(losses)

  clear_print("Starting script, max time left: %s minutes" % get_time_left(1,1, num_experiments))

  i = 0
  json_dir = "results/json/adult_losses"
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

  ylabel = "Accuracy %"
  plot_title = "Train accuracies for Adult Dataset with bound %s" % bound
  plot_dir = "results/plots/adult_losses/%s"
  plot_results(all_results, "train_accs", losses, 1, ylabel, plot_title, plot_dir % "train_acc")

  plot_title = "Test accuracies for Adult Dataset with bound %s" % bound
  plot_results(all_results, "test_accs", losses, 2, ylabel, plot_title, plot_dir % "test_acc")

  ylabel = "Runtime [s]"
  plot_title = "Runtimes for Adult Dataset with bound %s" % bound
  plot_results(all_results, "runtimes", losses, 3, ylabel, plot_title, plot_dir % "runtimes")

def run_experiments(loss, experiment, num_experiments):
  nn_results = {
    "train_accs": {},
    "val_accs": {},
    "test_accs": {},
    "runtimes": {},
    "objs": {}
  }
  i = 0
  for N in num_examples:
    i += 1
    nn_results["train_accs"][N] = []
    nn_results["val_accs"][N] = []
    nn_results["test_accs"][N] = []
    nn_results["runtimes"][N] = []
    nn_results["objs"][N] = []
    clear_print("Max time left: %s" % get_time_left(i, experiment, num_experiments))
    for s in seeds:
      data = load_data("adult", N, s)
      arch = get_architecture(data, [hl_neurons])

      clear_print("%s:  HL_Neurons: %s, N: %s, Seed: %s" % (loss, hl_neurons, N, s))
      nn = get_gurobi_nn(milps[loss], data, arch, bound)
      nn.train(60*time, focus)
      obj = nn.get_objective()
      runtime = nn.get_runtime()
      varMatrices = nn.extract_values()
      train_acc = infer_and_accuracy(nn.data["train_x"], nn.data["train_y"], varMatrices, nn.architecture)
      val_acc = infer_and_accuracy(nn.data["val_x"], nn.data["val_y"], varMatrices, nn.architecture)
      test_acc = infer_and_accuracy(nn.data["test_x"], nn.data["test_y"], varMatrices, nn.architecture)

      nn_results["train_accs"][N].append(train_acc)
      nn_results["val_accs"][N].append(val_acc)
      nn_results["test_accs"][N].append(test_acc)
      nn_results["runtimes"][N].append(runtime)
      nn_results["objs"][N].append(obj)

  return nn_results

def plot_results(all_results, result_type, losses, index, ylabel, title, plot_dir):
  x = num_examples
  plt.figure(index)
  colors = sns.color_palette("husl", len(losses))
  i = 0
  for loss in losses:
    y, err = get_mean_std(all_results[loss][result_type].values())
    plt.plot(x,y, label=loss, color = colors[i])
    plt.fill_between(x, y - err, y + err, alpha=0.3, facecolor=colors[i])
    i += 1
  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel(ylabel)
  plt.title(title)
  pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
  file_name = "Time:%s-HL_Neurons:%s-Bound:%s-S:%s" % (time, hl_neurons, bound, len(seeds))
  title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
  plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')
  #plt.show()

def get_mean_std(results):
  mean = np.array([np.mean(z) for z in results])
  std = np.array([np.std(z) for z in results])
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
