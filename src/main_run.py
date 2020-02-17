from milp.cplex_nn import get_cplex_nn
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.min_hinge_reg import MIN_HINGE_REG
from helper.misc import inference, calc_accuracy,clear_print
from helper.data import load_data, get_batches, load_heart, load_adult
from helper.save_data import DataSaver
import argparse
from keras.datasets import mnist,cifar10
import numpy as np
from pdb import set_trace

milps = {
  "min_w": MIN_W,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "min_hinge_reg": MIN_HINGE_REG
}

datasets = {
  "mnist": mnist,
  "cifar10": cifar10,
  "heart": None,
  "adult": None,
}

solvers = {
  "gurobi": get_gurobi_nn,
  "cplex": get_cplex_nn
}

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--solver', default="gurobi", type=str)
  parser.add_argument('--hl', default='16', type=str)
  parser.add_argument('--ex', default=10, type=int)
  parser.add_argument('--focus', default=0, type=int)
  parser.add_argument('--time', default=1, type=float)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--loss', default="min_w", type=str)
  parser.add_argument('--data', default="mnist", type=str)
  parser.add_argument('--bound', default=1, type=int)
  parser.add_argument('--batch', default=0, type=int)
  parser.add_argument('--save', action='store_true', help="An optional flag to save data")
  args = parser.parse_args()
  
  solver = args.solver
  hl = [int(x) for x in args.hl.split("-")]

  N = args.ex
  focus = args.focus
  train_time = args.time
  seed = args.seed
  loss = args.loss
  data = args.data
  bound = args.bound
  batch_size = args.batch

  print(args)

  if loss not in milps:
    raise Exception("MILP model %s not known" % loss)

  if data not in datasets:
    raise Exception("Dataset %s not known" % data)

  if solver not in solvers:
    raise Exception("Solver %s not known" % solver)


  # Load data (MNIST/CIFAR10)
  if data == 'heart':
    data = load_heart(N, seed)
  elif data == 'adult':
    data = load_adult(N, seed)
  else:
    data = load_data(datasets[data], N, seed)
  batches = [data]

  if batch_size > 0 and batch_size < N:
    batches = get_batches(data, batch_size)
  else:
    batch_size = N

  in_neurons = [data["train_x"].shape[1]]
  out_neurons = [data["oh_train_y"].shape[1]]

  # Set up NN layers, including input size, hidden layer sizes and output size
  architecture = in_neurons + hl + out_neurons
  print_str = "Architecture: %s. N: %s. Solver: %s. Loss: %s. Bound: %s"
  clear_print(print_str % ("-".join([str(x) for x in architecture]), len(data["train_x"]), solver, loss, bound))

  get_nn = solvers[solver]
  networks = []
  network_vars = []

  batch_num = 0
  for batch in batches:
    clear_print("Batch: %s. Examples: %s-%s" % (batch_num+1, batch_num*batch_size, (batch_num+1)*batch_size))
    nn = get_nn(milps[loss], batch, architecture, bound)
    nn.train(train_time*60, focus)

    obj = nn.get_objective()
    print("Objective value: ", obj)

    varMatrices = nn.extract_values()

    infer_train = inference(nn.data["train_x"], varMatrices, nn.architecture)
    infer_test = inference(nn.data["test_x"], varMatrices, nn.architecture)
    train_acc = calc_accuracy(infer_train, nn.data["train_y"])
    test_acc = calc_accuracy(infer_test, nn.data["test_y"])

    print("Training accuracy: %s " % (train_acc))
    print("Testing accuracy: %s " % (test_acc))

    networks.append(nn)
    network_vars.append(varMatrices)

    batch_num += 1

  if batch_size > 0 and batch_size != N:
    clear_print("Using warm start from batches")
    all_vars = network_vars[0]
    all_vars = {}
    mean_vars = {}
    bound_matrix = {}
    num_batches = len(network_vars)
    for key in network_vars[0]:
      all_vars[key] = np.stack([tmp[key] for tmp in network_vars])
      mean_vars[key] = np.mean(all_vars[key], axis=0)
      mean_vars[key][mean_vars[key] < 0] -= 1e-5
      mean_vars[key][mean_vars[key] >= 0] += 1e-5
      mean_vars[key] = np.round(mean_vars[key])
      if "w_" in key or "b_" in key:
        tmp_min = np.min(all_vars[key],axis=0)
        tmp_max = np.max(all_vars[key],axis=0)
        tmp_min[tmp_min > 0] = -bound
        tmp_max[tmp_max < 0] = bound
        bound_matrix["%s_%s" % (key,"lb")] = tmp_min
        bound_matrix["%s_%s" % (key,"ub")] = tmp_max

    warm_nn = get_nn(milps[loss], data, architecture, bound)
    #warm_nn.warm_start(mean_vars)
    warm_nn.update_bounds(bound_matrix)
    warm_nn.train(train_time*60, focus)

    obj = warm_nn.get_objective()
    print("Objective value: ", obj)

    varMatrices = warm_nn.extract_values()

    infer_train = inference(warm_nn.data["train_x"], varMatrices, warm_nn.architecture)
    infer_test = inference(warm_nn.data["test_x"], varMatrices, warm_nn.architecture)
    train_acc = calc_accuracy(infer_train, warm_nn.data["train_y"])
    warm_test_acc = calc_accuracy(infer_test, warm_nn.data["test_y"])

    print("Training accuracy: %s " % (train_acc))
    print("Testing accuracy: %s " % (warm_test_acc))

    clear_print("All data at once")
    nn = get_nn(milps[loss], data, architecture, bound)
    nn.train(train_time*60, focus)

    obj = nn.get_objective()
    print("Objective value: ", obj)

    varMatrices = nn.extract_values()

    infer_train = inference(nn.data["train_x"], varMatrices, nn.architecture)
    infer_test = inference(nn.data["test_x"], varMatrices, nn.architecture)
    train_acc = calc_accuracy(infer_train, nn.data["train_y"])
    test_acc = calc_accuracy(infer_test, nn.data["test_y"])

    print("Training accuracy: %s " % (train_acc))
    print("Testing accuracy: %s " % (test_acc))

    warm_nn_time = sum([i.get_runtime() for i in networks]) + warm_nn.get_runtime()

    clear_print("Warm start run time: %.2f. Accuracy: %s" % (warm_nn_time, warm_test_acc))
    clear_print("All at once run time: %.2f. Accuracy: %s" % (nn.get_runtime(), test_acc))

    infer_train = inference(nn.data["train_x"], mean_vars, nn.architecture)
    infer_test = inference(nn.data["test_x"], mean_vars, nn.architecture)
    train_acc = calc_accuracy(infer_train, nn.data["train_y"])
    test_acc = calc_accuracy(infer_test, nn.data["test_y"])

    clear_print("Training accuracy for mean parameters: %s" % (train_acc))
    clear_print("Testing accuracy for mean parameters: %s" % (test_acc))

  w1 = varMatrices['w_1']
  b1 = varMatrices['b_1']
  act1 = varMatrices['act_1']
  train = nn.data['train_x']
  set_trace()

  if args.save:
    if solver == 'gurobi':
      progress = nn.m._progress
    else:
      progress = nn.progress
    DS = DataSaver(nn, architecture, N, focus, solver)
    DS.plot_periodic(progress)
    DS.save_json(train_acc, test_acc)

