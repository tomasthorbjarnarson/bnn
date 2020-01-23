import os
import subprocess
from subprocess import PIPE
import re
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from datetime import datetime
from globals import ARCHITECTURES
from helper.misc import inference, calc_accuracy
from milp.gurobi_bnn import Gurobi_BNN as bnn

Icarte_dir = '../Icarte/bnn/src'

num_examples = {
  1: [1,2,3,4,5,6,7,8,9,10],
  2: [1,2,3,4,5],
  3: [1,2,3]
}

seeds = [1,2,3,4,5]

short = True
TIME = 15
if short:
  num_examples = {
    1: [1,2,3,4,5],
    2: [1,2,3],
    3: [1]
  }
  seeds = [1,2]
  TIME = 1

def clear_print(text):
  print("====================================")
  print(text)
  print("====================================")

def get_mean_std(results):
  mean = [np.mean(z[0]) for z in results]
  std = [np.std(z[0]) for z in results]
  return mean, std

def get_time_left(arch, example, experiment):
  num_experiments = 3
  num_ex = num_examples[arch]
  num_ex = num_ex[num_ex.index(example):]
  if arch == 1:
    examples_left = len(num_ex) + len(num_examples[2]) + len(num_examples[3])
  elif arch == 2:
    examples_left = len(num_ex) + len(num_examples[3])
  else:
    examples_left = len(num_ex)

  total_ex = (len(num_examples[1]) + len(num_examples[2]) + len(num_examples[3]))
  total_time = total_ex*len(seeds)*num_experiments*TIME

  ex_finished = total_ex*(experiment-1) + total_ex-examples_left

  return total_time - ex_finished*len(seeds)*TIME

def compare_test_accuracies():

  clear_print("Starting script, max time left: %s minutes" % get_time_left(1,1,1))

  print("Running min-w BNN experiments!")
  min_w_results = run_bnn_experiments("min_w", 1)
  print("Running max-acc BNN experiments!")  
  max_acc_results = run_bnn_experiments("max_acc", 2)
  print("Running GD experiments!")
  gd_results = run_gd_experiments(3)

  for i in min_w_results:
    x = [10*z for z in num_examples[i]]
    min_w_y, min_w_err = get_mean_std(min_w_results[i])
    max_acc_y, max_acc_err = get_mean_std(max_acc_results[i])
    gd_y, gd_err = get_mean_std(gd_results[i])
    plt.figure(i)
    plt.errorbar(x, min_w_y, yerr=min_w_err, capsize=1, label="Min-w test performance")
    plt.errorbar(x, max_acc_y, yerr=max_acc_err, capsize=1, label="Max-acc test performance")
    plt.errorbar(x, gd_y, yerr=gd_err, capsize=1, label="GD test performance")
    plt.legend()
    plt.xlabel("Number of examples")
    plt.ylabel("Test performance %")
    plt.title("Compare test accuracies")
    plot_dir = "results/plots/compare_test_accuracies"
    pathlib.Path(plot_dir).mkdir(exist_ok=True)
    title = "#HL:%s_Time:%s" % (i-1, datetime.now().strftime("%d %b %H:%M"))
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

    #plt.show()

def run_gd_experiments(experiment):
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
      clear_print("Max time left: %s" % get_time_left(i, N,experiment))
      for s in seeds:
        clear_print("GD:  hls: %s, N: %s, Seed: %s" % (hls, N, s))

        result = subprocess.run(run_str % (s, hls, N, TIME), shell=True, stdout=PIPE, stderr=PIPE)
        test_perf = re.findall(b"Test .*", result.stdout)[0]
        time = re.findall(b"= .*\[s]", result.stdout)[0]
        test_perf = float(test_perf[-4:])*100
        time = float(time[1:-3])
        acc.append(test_perf)
        runtime.append(time)
      gd_results[i].append((acc, runtime))

  os.chdir(current_dir)

  return gd_results

def run_bnn_experiments(obj, experiment):
  bnn_results = {}
  for i in ARCHITECTURES:
    bnn_results[i] = []
    arch = ARCHITECTURES[i]
    for N in num_examples[i]:
      acc = []
      runtime = []
      clear_print("Max time left: %s" % get_time_left(i, N,experiment))
      for s in seeds:
        clear_print("%s:  Arch: %s, N: %s, Seed: %s" % (obj, arch, N, s))
        Gurobi_BNN = bnn(N*10, arch, obj, s, False)
        Gurobi_BNN.train(60*TIME, 0)
        Gurobi_obj = Gurobi_BNN.get_objective()
        Gurobi_runtime = Gurobi_BNN.get_runtime()
        varMatrices = Gurobi_BNN.extract_values()

        infer_test = inference(Gurobi_BNN.test_x, varMatrices, Gurobi_BNN.architecture)
        test_acc = calc_accuracy(infer_test, Gurobi_BNN.test_y)
        acc.append(test_acc)
        runtime.append(Gurobi_runtime)

      bnn_results[i].append((acc, runtime))

  return bnn_results
