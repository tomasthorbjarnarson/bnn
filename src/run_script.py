from scripts.compare_gurobi_cplex import compare_gurobi_cplex
from scripts.compare_test_accuracies import compare_test_accuracies
from scripts.compare_heart_one_hl import compare_heart_one_hl
from time import time
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--script', type=str)
  parser.add_argument('--losses', default="", type=str)
  parser.add_argument('--save', action='store_true')
  args = parser.parse_args()
  script = args.script
  save = args.save
  losses = args.losses.split(",")
  print(losses)

  start = time()
  if script == 'compare_gurobi_cplex':
    compare_gurobi_cplex()
  elif script == 'compare_test_accuracies':
    compare_test_accuracies(losses, save)
  elif script == 'compare_heart_one_hl':
    compare_heart_one_hl(losses, save)
  else:
    raise Exception("Script %s not known" % script)
  end = time()
  print("Time to run script %s:  %.2f" % (script, (end - start)))

