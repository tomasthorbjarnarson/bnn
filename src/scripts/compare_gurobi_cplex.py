from milp.gurobi_bnn import get_gurobi_bnn
from milp.cplex_bnn import get_cplex_bnn
from milp.max_correct_bnn import MAX_CORRECT_BNN
from helper.misc import clear_print
from datetime import datetime
from globals import ARCHITECTURES
import matplotlib.pyplot as plt
import pathlib

numExamples = {
  1: [10,20,30,40,50,60,70,80,90,100],
  2: [10, 20, 30, 40, 50],
  3: [10, 20, 30]
}

times = {
  1: 3,
  2: 5,
  3: 10
}

focuses = {
  1: 3,
  2: 3,
  3: 0
}
short = False
if short:
  numExamples = {
    1: [5, 10, 15],
    2: [1,2],
    3: [1,2]
  }


def compare_gurobi_cplex():
  ARCHITECTURES.pop(1)
  for i in ARCHITECTURES:
    arch = ARCHITECTURES[i]
    time = times[i]
    examples = numExamples[i]
    Gurobi_results = {}
    Cplex_results = {}
    for f in range(4):
      Gurobi_results[f] = []
      Cplex_results[f] = []
      for N in examples:
        clear_print("Focus: %s. Architecture: %s. # Examples: %s" % (f, arch, N))
        Gurobi_BNN = get_gurobi_bnn(MAX_CORRECT_BNN, N, arch, seed=10)
        Cplex_BNN = get_cplex_bnn(MAX_CORRECT_BNN, N, arch, seed=10)

        Gurobi_BNN.train(60*time, f)
        Cplex_BNN.train(60*time, f)

        Gurobi_obj = Gurobi_BNN.get_objective()
        Gurobi_runtime = Gurobi_BNN.get_runtime()

        Cplex_obj = Cplex_BNN.get_objective()
        Cplex_runtime = Cplex_BNN.get_runtime()

        Gurobi_results[f].append((Gurobi_obj, Gurobi_runtime))
        print("Gurobi", (Gurobi_obj, Gurobi_runtime))
        Cplex_results[f].append((Cplex_obj, Cplex_runtime))
        print("Cplex", (Cplex_obj, Cplex_runtime))

    x = examples
    plt.figure(i)
    for f in range(4):
      Gurobi_y = [z[1] for z in Gurobi_results[f]]
      plt.plot(x, Gurobi_y, label="Gurobi runtime, focus: %s" % f, color=((f+1)*0.25,0,0,1))
    for f in range(4):
      Cplex_y = [z[1] for z in Cplex_results[f]]
      plt.plot(x, Cplex_y, label="Cplex runtime, focus: %s" % f, color=(0,(f+1)*0.25,0,1))
      #for j in range(len(Gurobi_results[f])):
      #  plt.annotate(Gurobi_results[f][j][0], (x[j], Gurobi_y[j]))
      #  plt.annotate(int(Cplex_results[f][j][0]), (x[j], Cplex_y[j]))
    plt.legend()
    plt.xlabel("Number of examples")
    plt.ylabel("Runtime [s]")
    plt.title("Gurobi vs Cplex for max-correct-bnn for %s hidden layers" % (i-1))
    #plt.show()
    plot_dir = "results/plots/compare_gurobi_cplex"
    pathlib.Path(plot_dir).mkdir(exist_ok=True)
    title = "#HL:%s_Time:%s" % (i-1, datetime.now().strftime("%d %b %H:%M"))
    #plt.show()
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')
