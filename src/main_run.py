from milp.cplex_nn import get_cplex_nn
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.min_hinge_reg import MIN_HINGE_REG
from helper.misc import inference, calc_accuracy,clear_print
from helper.data import load_data
from helper.save_data import DataSaver
import argparse
from keras.datasets import mnist,cifar10
import time

milps = {
  "min_w": MIN_W,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "min_hinge_reg": MIN_HINGE_REG
}

datasets = {
  "mnist": mnist,
  "cifar10": cifar10
}

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--solver', default="gurobi", type=str)
  parser.add_argument('--hl', default=16, type=str)
  parser.add_argument('--ex', default=3, type=int)
  parser.add_argument('--focus', default=0, type=int)
  parser.add_argument('--time', default=1, type=float)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--loss', default="min_w", type=str)
  parser.add_argument('--data', default="mnist", type=str)
  parser.add_argument('--bound', default=1, type=int)
  parser.add_argument('--save', action='store_true', help="An optional flag to save data")
  args = parser.parse_args()
  
  solver = args.solver
  hl = [int(x) for x in args.hl.split("-")]

  numExamples = args.ex
  focus = args.focus
  train_time = args.time
  seed = args.seed
  loss = args.loss
  data = args.data
  bound = args.bound

  print(args)

  if loss not in milps:
    raise Exception("MILP model %s not known" % loss)

  if data not in datasets:
    raise Exception("Dataset %s not known" % data)

  # Load data (MNIST/CIFAR10)
  data = load_data(datasets[data], numExamples, seed)

  # Set up NN layers, including input size, hidden layer sizes and output size
  architecture = [data["train_x"].shape[1]] + hl + [data["oh_train_y"].shape[1]]
  print_str = "Architecture: %s. N: %s. Solver: %s. Loss: %s. Bound: %s"
  clear_print(print_str % ("-".join([str(x) for x in architecture]), len(data["train_x"]), solver, loss, bound))


  start = time.time()
  if solver == 'gurobi':
    nn = get_gurobi_nn(milps[loss], data, architecture, bound)
  elif solver =='cplex':
    nn = get_cplex_nn(milps[loss], data, architecture, bound)
  else:
    raise Exception("Solver %s not known" % solver)
  end = time.time()
  clear_print("Time to init params: %s" % ((end - start)))

  nn.train(train_time*60, focus)

  obj = nn.get_objective()
  print("Objective value: ", obj)

  varMatrices = nn.extract_values()

  tr_time = time.time()

  infer_train = inference(nn.data["train_x"], varMatrices, nn.architecture)
  print("Infer train time: %s" % (time.time() - tr_time))
  te_time = time.time()
  infer_test = inference(nn.data["test_x"], varMatrices, nn.architecture)
  print("Infer test time: %s" % (time.time() - te_time))

  train_acc = calc_accuracy(infer_train, nn.data["train_y"])
  test_acc = calc_accuracy(infer_test, nn.data["test_y"])

  print("Training accuracy: %s " % (train_acc))
  print("Testing accuracy: %s " % (test_acc))

  w1 = varMatrices['w_1']
  b1 = varMatrices['b_1']
  act1 = varMatrices['act_1']

  from pdb import set_trace
  set_trace()

  if args.save:
    if solver == 'gurobi':
      progress = nn.m._progress
    else:
      progress = nn.progress
    DS = DataSaver(nn, architecture, numExamples, focus, solver)
    DS.plot_periodic(progress)
    DS.save_json(train_acc, test_acc)

