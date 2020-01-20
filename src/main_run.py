from milp.cplex_bnn import Cplex_BNN
from milp.gurobi_bnn import Gurobi_BNN
from helper.misc import inference, calc_accuracy
from helper.save_data import DataSaver
from helper.mnist import imshow
from globals import ARCHITECTURES
import argparse
import numpy as np


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--solver', default="gurobi", type=str)
  parser.add_argument('--arch', default=2, type=int)
  parser.add_argument('--ex', default=3, type=int)
  parser.add_argument('--focus', default=3, type=int)
  parser.add_argument('--time', default=1, type=float)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--obj', default="min_w", type=str)
  parser.add_argument('--save', action='store_true', help="An optional flag to save data")
  args = parser.parse_args()
  
  solver = args.solver
  architecture = ARCHITECTURES[args.arch]
  numExamples = args.ex
  focus = args.focus
  time = args.time
  seed = args.seed
  obj = args.obj

  print(args)

  if solver == 'gurobi':
    bnn = Gurobi_BNN(numExamples, architecture, obj, seed)
  elif solver =='cplex':
    bnn = Cplex_BNN(numExamples, architecture, obj, seed)
  else:
    raise Exception("Solver %s not known" % solver)
  bnn.train(time*60, focus)

  obj = bnn.get_objective()
  print("Objective value: ", obj)

  varMatrices = bnn.extract_values()

  if False:
    tmp = np.abs(varMatrices['w_1'])
    tmp = np.sum(tmp, axis=1)
    imshow(tmp)

  infer_train = inference(bnn.train_x, varMatrices, bnn.architecture)
  infer_test = inference(bnn.test_x, varMatrices, bnn.architecture)

  train_acc = calc_accuracy(infer_train, bnn.train_y)
  test_acc = calc_accuracy(infer_test, bnn.test_y)

  print("Training accuracy: %s " % (train_acc))
  print("Testing accuracy: %s " % (test_acc))

  if args.save:
    if solver == 'gurobi':
      progress = bnn.m._progress
    else:
      progress = bnn.progress
    DS = DataSaver(bnn, architecture, numExamples, focus, solver)
    DS.plot_periodic(progress)
    DS.save_json(train_acc, test_acc)

