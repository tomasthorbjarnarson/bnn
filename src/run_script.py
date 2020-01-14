from scripts.compare_gurobi_cplex import compare
from time import time

if __name__ == '__main__':
  start = time()
  compare()
  end = time()
  print("Time to run script compare %.2f" % (end - start))

