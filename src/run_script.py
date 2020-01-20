from scripts.compare_gurobi_cplex import compare_gurobi_cplex
from scripts.compare_test_accuracies import compare_test_accuracies
from time import time
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--script', type=str)
  args = parser.parse_args()
  script = args.script

  start = time()
  if script == 'compare_gurobi_cplex':
    compare_gurobi_cplex()
  elif script == 'compare_test_accuracies':
    compare_test_accuracies()
  else:
    raise Exception("Script %s not known" % script)
  end = time()
  print("Time to run script %s:  %.2f" % (script, (end - start)))

