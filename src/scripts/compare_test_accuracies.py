import os
import subprocess
from subprocess import PIPE
import re
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
from datetime import datetime
from globals import ARCHITECTURES
from helper.misc import inference, calc_accuracy, clear_print
from helper.data import load_data
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.sat_margin import SAT_MARGIN

Icarte_dir = '../Icarte/bnn/src'

num_examples = {
  1: [1,2,3,4,5,6,7,8,9,10],
  2: [1,2,3,4,5],
  3: [1,2,3]
}

times = {
  1: 15,
  2: 90,
  3: 90
}
seeds = [1,2,3]

milps = {
  "min_w": MIN_W,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "sat_margin": SAT_MARGIN
}


short = False
if short:
  num_examples = {
    1: [1,2,3],
    2: [1,2],
    3: [1]
  }
  seeds = [1,2]
  times = {
    1: 1,
    2: 1,
    3: 1
  }

def compare_test_accuracies(losses, plot=False):

  all_results = {k: [] for k in losses}
  all_results = {}
  num_experiments = len(losses)

  clear_print("Starting script, max time left: %s minutes" % get_time_left(1,1,1, num_experiments))

  i = 0
  json_dir = "results/json/compare_test_accuracies"
  pathlib.Path(json_dir).mkdir(exist_ok=True)

  for loss in losses:
    i += 1
    print("Running %s nn experiments!" % loss)
    timestr = "%s-%s-%s" % (times[1], times[2], times[3])
    file_name = "%s-Times:%s_S:%s" % (loss, timestr, len(seeds))
    json_path = "%s/%s.json" % (json_dir, file_name)
    if pathlib.Path(json_path).is_file():
      print("Path %s exists" % json_path)
      with open(json_path, "r") as f:
        data = json.loads(f.read())
        tmp = data["results"]
        all_results[loss] = {}
        for k in tmp:
          all_results[loss][int(k)] = tmp[k]
      continue

    if loss in milps:
      all_results[loss] = run_experiments(loss, i, num_experiments)
    elif loss == "gd":
      all_results["gd"] = run_gd_experiments(i, num_experiments)
    else:
      print("Loss %s unknown" % loss)
      continue
    with open(json_path, "w") as f:
      data = {"results": all_results[loss], "ts": datetime.now().strftime("%d-%m-%H:%M")}
      json.dump(data, f)

  for i in all_results[losses[0]]:
    x = [10*z for z in num_examples[i]]
    plt.figure(i)
    for loss in losses:
      y, err = get_acc_mean_std(all_results[loss][i])
      plt.errorbar(x, y, yerr=err, capsize=3, label="%s test performance" % loss)
    plt.legend()
    plt.xlabel("Number of examples")
    plt.ylabel("Test performance %")
    plt.title("Compare test accuracies")
    plot_dir = "results/plots/compare_test_accuracies/%s" % datetime.now().strftime("%d-%m-%H:%M")
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
    title = "#HL:%s-Time:%s-S:%s" % (i-1, times[i], len(seeds))
    if plot:
      plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  for i in all_results[losses[0]]:
    x = [10*z for z in num_examples[i]]
    plt.figure(i*10)
    for loss in losses:
      y, err = get_runtime_mean_std(all_results[loss][i])
      plt.errorbar(x, y, yerr=err, capsize=3, label="%s runtime" % loss)
    plt.legend()
    plt.xlabel("Number of examples")
    plt.ylabel("Runtime [s]")
    plt.title("Compare runtimes")
    plot_dir = "results/plots/compare_loss_runtimes/%s" % datetime.now().strftime("%d-%m-%H:%M")
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
    title = "#HL:%s-Time:%s-S:%s" % (i-1, times[i], len(seeds))
    if plot:
      plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

def run_gd_experiments(experiment, num_experiments):
  current_dir = os.getcwd()
  os.chdir(Icarte_dir)
  gd_results = {}
  run_str = 'python run.py --model="gd_b" --lr=1e-3 --seed=%s --hls=%s --exs=%s --ins=0 --to=%s'
  for i in ARCHITECTURES:
    gd_results[i] = []
    hls = i-1
    for N in num_examples[i]:
      acc = []
      runtime = []
      clear_print("Max time left: %s" % get_time_left(i, N, experiment, num_experiments))
      for s in seeds:
        clear_print("GD:  hls: %s, N: %s, Seed: %s" % (hls, N, s))

        result = subprocess.run(run_str % (s, hls, N, times[i]), shell=True, stdout=PIPE, stderr=PIPE)
        test_perf = re.findall(b"Test .*", result.stdout)[0]
        time = re.findall(b"= .*\[m]", result.stdout)[0]
        test_perf = float(test_perf[-4:])*100
        time = float(time[1:-3])*60
        acc.append(test_perf)
        runtime.append(time)
      gd_results[i].append((acc, runtime))

  os.chdir(current_dir)

  return gd_results

def run_experiments(loss, experiment, num_experiments):
  nn_results = {}
  
  for i in ARCHITECTURES:
    nn_results[i] = []
    arch = ARCHITECTURES[i]
    for N in num_examples[i]:
      accs = []
      runtimes = []
      clear_print("Max time left: %s" % get_time_left(i, N, experiment, num_experiments))
      for s in seeds:
        data = load_data("mnist", N*10, s)
        clear_print("%s:  Arch: %s, N: %s, Seed: %s" % (loss, arch, N*10, s))
        nn = get_gurobi_nn(milps[loss],data, arch, 1)
        nn.train(60*times[i], 0)
        obj = nn.get_objective()
        runtime = nn.get_runtime()
        varMatrices = nn.extract_values()

        infer_test = inference(nn.data["test_x"], varMatrices, nn.architecture)
        test_acc = calc_accuracy(infer_test, nn.data["test_y"])
        accs.append(test_acc)
        runtimes.append(runtime)

      nn_results[i].append((accs, runtimes))

  return nn_results


def get_acc_mean_std(results):
  mean = [np.mean(z[0]) for z in results]
  std = [np.std(z[0]) for z in results]
  return mean, std

def get_runtime_mean_std(results):
  mean = [np.mean(z[1]) for z in results]
  std = [np.std(z[1]) for z in results]
  return mean, std

def get_time_left(arch, example, experiment, num_experiments):
  time_left = 0

  for j in range(num_experiments):
    if j == experiment - 1:
      for i in num_examples:
        if i == arch :
          for k in num_examples[i][example-1:]:
            time_left += times[i]*len(seeds)
        elif i > arch:
          for k in num_examples[i]:
            time_left += times[i]*len(seeds)
    elif j > experiment - 1:
      for i in num_examples:
        for k in num_examples[i]:
            time_left += times[i]*len(seeds)

  days = time_left // (60*24)
  time_left -= days*60*24
  hours = time_left // 60
  time_left -= hours*60
  minutes = time_left % 60

  return "%s days, %s hours, %s minutes" % (days, hours, minutes)
