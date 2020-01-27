from milp.gurobi_bnn import get_gurobi_bnn
from milp.cplex_bnn import get_cplex_bnn
from milp.min_w_bnn import MIN_W_BNN
from datetime import datetime
from globals import ARCHITECTURES
import matplotlib.pyplot as plt
import pathlib

numExamples = {
  1: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
  2: [1, 2, 3, 4, 5],
  3: [1, 2, 3]
}

times = {
  1: 30,
  2: 90,
  3: 150
}

focuses = {
  1: 3,
  2: 3,
  3: 0
}

log = True

def compare_gurobi_cplex():
  for i in ARCHITECTURES:
    arch = ARCHITECTURES[i]
    time = times[i]
    focus = focuses[i]
    examples = numExamples[i]
    Gurobi_results = []
    Cplex_results = []
    for N in examples:
      print("Architecture: %s. # Examples: %s" % (arch, N))
      Gurobi_BNN = get_gurobi_bnn(MIN_W_BNN, N, arch, log=log)
      Cplex_BNN = get_cplex_bnn(MIN_W_BNN, N, arch, log=log)

      Gurobi_BNN.train(60*time, focus)
      Cplex_BNN.train(60*time, focus)

      Gurobi_obj = Gurobi_BNN.get_objective()
      Gurobi_runtime = Gurobi_BNN.get_runtime()

      Cplex_obj = Cplex_BNN.get_objective()
      Cplex_runtime = Cplex_BNN.get_runtime()

      Gurobi_results.append((Gurobi_obj, Gurobi_runtime))
      print("Gurobi", (Gurobi_obj, Gurobi_runtime))
      Cplex_results.append((Cplex_obj, Cplex_runtime))
      print("Cplex", (Cplex_obj, Cplex_runtime))

    x = examples
    Gurobi_y = [z[1] for z in Gurobi_results]
    Cplex_y = [z[1] for z in Cplex_results]

    plt.figure(i)
    plt.plot(x, Gurobi_y, label="Gurobi runtime")
    plt.plot(x, Cplex_y, label="Cplex runtime")
    for j in range(len(Gurobi_results)):
      plt.annotate(Gurobi_results[j][0], (x[j], Gurobi_y[j]))
      plt.annotate(int(Cplex_results[j][0]), (x[j], Cplex_y[j]))
    plt.legend()
    plt.xlabel("Number of examples")
    plt.ylabel("Runtime [s]")
    plt.title("Gurobi vs Cplex for 0 hidden layers")
    #plt.show()
    plot_dir = "results/plots/compare_gurobi_cplex"
    pathlib.Path(plot_dir).mkdir(exist_ok=True)
    title = "#HL:%s-Focus:%s_Time:%s" % (i-1, focus, datetime.now().strftime("%d %b %H:%M"))
    plt.savefig("%s/%s.png" % (plot_dir,title),  bbox_inches='tight')
