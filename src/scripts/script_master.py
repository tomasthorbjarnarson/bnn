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
from milp.max_m import MAX_M
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.sat_margin import SAT_MARGIN
from gd.gd_nn import GD_NN

focus = 1

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


class Script_Master():
  def __init__(self, script_name, losses, dataset, num_examples, max_runtime,
               seeds, hls, bounds, regs=[], lr=1e-3, show=True):
    self.script_name = script_name
    self.losses = losses
    for loss in self.losses:
      if loss not in milps and loss not in gds:
        raise Exception("Loss %s not known" % loss)

    self.dataset = dataset
    self.num_examples = num_examples
    self.max_runtime = max_runtime
    self.seeds = seeds
    self.bounds = bounds
    self.regs = regs
    if len(bounds) == 1:
      self.bound = bounds[0]
    elif len(losses) == 1 and len(bounds) > 1:
      self.losses = []
      for bound in bounds:
        self.losses.append("%s-bound=%s" % (losses[0], bound))
    elif len(losses) == 1 and len(regs) > 0:
      self.losses = []
      for reg in regs:
        self.losses.append("%s-reg=%s" % (losses[0],reg))
    else:
      raise Exception("Losses %s, Bounds %s and Regs %s incompatible" % (losses,bounds,regs))

    self.lr = lr
    self.show = show

    self.results = {}
    self.json_names = {}
    self.plot_names = {}
    self.hls = {}

    for hl in hls:
      hl_key = '-'.join([str(x) for x in hl])
      self.results[hl_key] = {}
      self.json_names[hl_key] = {}
      self.hls[hl_key] = hl

    self.max_time_left = len(self.losses)*len(num_examples)*len(seeds)*len(hls)*max_runtime

    for hl_key in self.json_names:
      if len(bounds) == 1:
        name = "Time:%s_HLs:%s_|S|:%s_Prec:%s" % (max_runtime, hl_key, len(seeds), self.bound)
      else:
        name = "Time:%s_HLs:%s_|S|:%s_|Prec|:%s" % (max_runtime, hl_key, len(seeds), len(bounds))

      for loss in self.losses:
        self.json_names[hl_key][loss] = "%s-%s" % (loss, name)
      self.plot_names[hl_key] = name

    self.json_dir = "results/json/%s_%s" % (script_name, dataset)
    self.plot_dir = "results/plots/%s_%s" % (script_name, dataset)
    pathlib.Path(self.json_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(self.plot_dir).mkdir(parents=True, exist_ok=True)


  def run_all(self):
    for hl_key in self.json_names:
      for loss in self.losses:
        json_path = "%s/%s.json" % (self.json_dir, self.json_names[hl_key][loss])
        if pathlib.Path(json_path).is_file():
          print("Path %s exists" % json_path)
          with open(json_path, "r") as f:
            data = json.loads(f.read())
            self.results[hl_key][loss] = data["results"]
        else:
          self.run_experiment(hl_key, loss)
          with open(json_path, "w") as f:
            data = {"results": self.results[hl_key][loss], "ts": datetime.now().strftime("%d-%m-%H:%M")}
            json.dump(data, f)

  def plot_all(self):
    settings = ["train_accs", "test_accs", "runtimes"]
    i = 1
    for hl_key in self.results:
      for setting in settings:
        self.plot_results(hl_key, setting, i)
        i += 1

  def run_experiment(self, hl_key, loss):
    nn_results = {
      "train_accs": {},
      "val_accs": {},
      "test_accs": {},
      "runtimes": {},
      "objs": {}
    }
    og_loss = loss
    # If there are multiple precisions for the same loss
    if "-bound=" in og_loss:
      loss, bound = og_loss.split("-bound=")
      bound = int(bound)
    else:
      bound = self.bound

    if "-reg=" in og_loss:
      loss,reg = og_loss.split("-reg=")
      reg = float(reg)
    else:
      reg = 0
    # Make a copy
    num_examples = list(self.num_examples)
    for N in num_examples:
      nn_results["train_accs"][N] = []
      nn_results["val_accs"][N] = []
      nn_results["test_accs"][N] = []
      nn_results["runtimes"][N] = []
      nn_results["objs"][N] = []
      nn_results["HL"] = []
      optimal_reached = []
      self.print_max_time_left()
      for s in self.seeds:
        clear_print("%s:  HLs: %s. N: %s. Seed: %s. Bound: %s. Reg: %s" % (loss, hl_key, N, s, bound, reg))
        data = load_data(self.dataset, N, s)
        arch = get_architecture(data, self.hls[hl_key])
        if loss in milps:
          nn = get_gurobi_nn(milps[loss], data, arch, bound, reg)
          nn.train(60*self.max_runtime, focus)
        else:
          nn = GD_NN(data, N, arch, self.lr, bound, s)
          nn.train(60*self.max_runtime)
        obj = nn.get_objective()
        runtime = nn.get_runtime()
        varMatrices = nn.extract_values()
        train_acc = infer_and_accuracy(nn.data["train_x"], nn.data["train_y"], varMatrices, nn.architecture)
        val_acc = infer_and_accuracy(nn.data["val_x"], nn.data["val_y"], varMatrices, nn.architecture)
        test_acc = infer_and_accuracy(nn.data["test_x"], nn.data["test_y"], varMatrices, nn.architecture)

        optimal_reached.append(obj <= nn.cutoff)
        clear_print("Runtime was: %s" % (runtime))
        print("")

        nn_results["train_accs"][N].append(train_acc)
        nn_results["val_accs"][N].append(val_acc)
        nn_results["test_accs"][N].append(test_acc)
        nn_results["runtimes"][N].append(runtime)
        nn_results["objs"][N].append(obj)
        if reg:
          hl = [int(v.sum()) for (k,v) in varMatrices.items() if "H_" in k]
          nn_results["HL"].append(sum(hl))
        else:
          nn_results["HL"].append(sum(arch[1:-1]))

        self.max_time_left -= self.max_runtime

      if self.script_name == "push" and any(optimal_reached):
        num_examples.append(N+num_examples[0])
        if self.max_time_left <= 0:
          self.max_time_left = self.max_runtime*len(self.seeds)

    self.results[hl_key][og_loss] = nn_results

  def plot_results(self, hl_key, setting, index):
    plt.figure(index)
    colors = sns.color_palette("husl", len(self.losses))
    for i,loss in enumerate(self.losses):
      x = [int(z) for z in self.results[hl_key][loss][setting].keys()]
      y, err = get_mean_std(self.results[hl_key][loss][setting].values())
      plt.plot(x,y, label=loss, color = colors[i])
      plt.fill_between(x, y - err, y + err, alpha=0.3, facecolor=colors[i])

    if len(self.bounds) == 1:
      title = "%s for %s dataset with bound %s" % (titles[setting], self.dataset, self.bound)
    else:
      title = "%s for %s dataset with different bounds" % (titles[setting], self.dataset)

    plt.legend()
    plt.xlabel("Number of examples")
    plt.ylabel(self.get_plot_ylabel(setting))
    plt.title(self.get_plot_title(setting))
    plt.ylim(self.get_plot_ylim(setting))

    plot_dir = "%s/%s" % (self.plot_dir, setting)
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
    file_name = self.plot_names[hl_key]
    title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
    if self.show:
      plt.show()
    else:
      plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  def subplot_results(self):
    settings = ["train_accs", "test_accs", "runtimes"]   
    j = 0
    for hl_key in self.results:
      fig, axs = plt.subplots(2,2)
      axs = axs.flatten()
      for setting in settings:
        colors = sns.color_palette("husl", len(self.losses))
        for i,loss in enumerate(self.losses):
          x = [int(z) for z in self.results[hl_key][loss][setting].keys()]
          y, err = get_mean_std(self.results[hl_key][loss][setting].values())
          axs[j].plot(x,y, label=loss, color = colors[i])
          axs[j].fill_between(x, y - err, y + err, alpha=0.3, facecolor=colors[i])
        axs[j].set_xlabel("Number of examples")
        axs[j].set_ylabel(self.get_plot_ylabel(setting))
        axs[j].set_title(self.get_plot_title(setting))
        axs[j].set_ylim(self.get_plot_ylim(setting))
        j += 1

      handles, labels = axs[0].get_legend_handles_labels()
      axs[-1].axis('off')
      axs[-1].legend(handles, labels, loc='upper left')

      plot_dir = "%s/%s" % (self.plot_dir, setting)
      pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
      file_name = self.plot_names[hl_key]
      title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
      if self.show:
        plt.show()
      else:
        plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  def plot_reg_results(self):
    def get_reg_label(reg):
      if reg == 0:
        return "No regularization"
      elif reg == -1:
        return "Hierarchical regularization"
      else:
        return "Weighted regularization, alpha=%s" % reg
    settings = ["train_accs", "test_accs", "runtimes"]
    markers = ['o', 'v', '+', '*', 'P', '^', 'v'][0:len(self.losses)]
    j = 0
    for hl_key in self.results:
      fig, axs = plt.subplots(2,2, figsize=(12,10))
      axs = axs.flatten()
      for setting in settings:
        colors = sns.color_palette("husl", len(self.losses))
        for i,loss in enumerate(self.losses):
          _,reg = loss.split("-reg=")
          reg = float(reg)
          x = self.results[hl_key][loss]["HL"]
          y = list(self.results[hl_key][loss][setting].values())
          axs[j].scatter(x,y, label=get_reg_label(reg), color=colors[i], marker=markers[i])
          if reg == 0 and setting == "test_accs":
            x = [0,np.max(x)]
            y = [np.min(y),np.min(y)]
            axs[j].plot(x,y, color=colors[i], linestyle="--")
        axs[j].set_xlabel("Number of neurons in hidden layer(s)")
        axs[j].set_ylabel(self.get_plot_ylabel(setting))
        axs[j].set_title(self.get_plot_title(setting))
        axs[j].set_ylim(self.get_plot_ylim(setting))
        j += 1
      handles, labels = axs[0].get_legend_handles_labels()
      axs[-1].axis('off')
      axs[-1].legend(handles, labels, loc='upper left')

      plot_dir = "%s" % (self.plot_dir)
      pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
      file_name = self.plot_names[hl_key]
      title = "%s_TS:%s" % (file_name, datetime.now().strftime("%d-%m-%H:%M"))
      if self.show:
        plt.show()
      else:
        plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')

  def print_max_time_left(self):
    time_left = self.max_time_left
    days = time_left // (60*24)
    time_left -= days*60*24
    hours = time_left // 60
    time_left -= hours*60
    minutes = time_left % 60

    clear_print("Max time left: %s days, %s hours, %s minutes" % (days, hours, minutes))

  def get_plot_title(self, setting):
    titles = {
      "train_accs": "Train",
      "test_accs": "Test",
      "runtimes": "Runtime"
    }
    return titles[setting]

  def get_plot_ylabel(self, setting):
    ylabels = {
      "train_accs": "Accuracy %",
      "test_accs": "Accuracy %",
      "runtimes": "Runtime [s]"
    }
    return ylabels[setting]

  def get_plot_ylim(self, setting):
    ylims = {
      "train_accs": [0,100],
      "test_accs": [0,100],
      "runtimes": [0, self.max_runtime*60]
    }
    return ylims[setting]

def get_mean_std(results):
  mean = np.array([np.mean(z) for z in results])
  std = np.array([np.std(z) for z in results])
  return mean, std
