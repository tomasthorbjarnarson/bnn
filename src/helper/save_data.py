from helper.NumpyEncoder import NumpyEncoder
from datetime import datetime
import matplotlib.pyplot as plt
import json
import math
import pathlib

def get_file_locations(architecture):
  arch_str = '-'.join([str(z) for z in architecture])
  plot_dir = "results/plots/%s" % arch_str
  json_dir = "results/json/%s" % arch_str
  pathlib.Path(plot_dir).mkdir(exist_ok=True)
  pathlib.Path(json_dir).mkdir(exist_ok=True)
  return plot_dir, json_dir

class DataSaver:
  def __init__(self, bnn, architecture, num_examples, focus, time):
    self.bnn = bnn
    self.architecture = architecture
    self.num_examples = num_examples
    self.focus = focus
    self.time_elapsed = math.floor(bnn.m.Runtime)

    now = datetime.now()
    nowStr = now.strftime("%d %b %H:%M")
    self.title = '#Exs:%s-Time:%s-Focus:%s_%s' % (num_examples,self.time_elapsed,focus,nowStr)
    self.plot_dir, self.json_dir = get_file_locations(architecture)

  def plot_periodic(self):
    per_filtered = [z for z in self.bnn.m._periodic if z[4] < 0.95]
    x = [z[3] for z in per_filtered]
    y = [z[1] for z in per_filtered]
    y2 = [z[2] for z in per_filtered]

    plt.plot(x,y, label="Best objective")
    plt.plot(x,y2, label="Best bound")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Sum of absolute weights")
    plt.title(self.title)
    last_acc = 0
    for sol in per_filtered:
      if sol[5] != last_acc:
        plt.annotate("%.2f" % sol[5],(sol[3], sol[1]))
        last_acc = sol[5]
    plt.show()
    plt.savefig("%s/%s.png" % (self.plot_dir, self.title), bbox_inches='tight')


  def save_json(self, train_acc, test_acc):
    now = datetime.now()
    nowStr = now.strftime("%d/%m/%Y %H:%M:%S")

    save = {
      'datetime': nowStr,
      'architecture': self.architecture,
      'num_examples': self.num_examples,
      'obj': self.bnn.m.ObjVal,
      'bound': self.bnn.m.ObjBound,
      'gap': self.bnn.m.MIPGap,
      'nodecount': self.bnn.m.NodeCount,
      'time': self.time_elapsed,
      'MIPFocus': self.focus,
      'trainingAcc': train_acc,
      'testingAcc': test_acc,
      'num_vars': self.bnn.m.NumVars,
      'num_int_vars': self.bnn.m.NumIntVars,
      'num_binary_vars': self.bnn.m.NumBinVars,
      'num_constrs': self.bnn.m.NumConstrs,
      'num_nonzeros': self.bnn.m.NumNZs,
      'periodic': self.bnn.m._periodic,
      'variables': self.bnn.extract_values(),
    }

    with open('%s/%s.json' % (self.json_dir, self.title), 'w') as f:
      json.dump(save, f, cls=NumpyEncoder)