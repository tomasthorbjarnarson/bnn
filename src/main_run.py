from milp.cplex_nn import get_cplex_nn
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.max_margin import MAX_MARGIN
from helper.misc import infer_and_accuracy, clear_print, get_bound_matrix,get_alt_bound_matrix,get_mean_vars
from helper.data import load_data, get_batches, get_architecture
from helper.save_data import DataSaver
import argparse
import numpy as np
from pdb import set_trace

milps = {
  "min_w": MIN_W,
  "max_correct": MAX_CORRECT,
  "min_hinge": MIN_HINGE,
  "max_margin": MAX_MARGIN
}

solvers = {
  "gurobi": get_gurobi_nn,
  "cplex": get_cplex_nn
}

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--solver', default="gurobi", type=str)
  parser.add_argument('--hls', default='16', type=str)
  parser.add_argument('--ex', default=10, type=int)
  parser.add_argument('--focus', default=0, type=int)
  parser.add_argument('--time', default=1, type=float)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--loss', default="min_w", type=str)
  parser.add_argument('--data', default="mnist", type=str)
  parser.add_argument('--bound', default=1, type=int)
  parser.add_argument('--batch', default=0, type=int)
  parser.add_argument('--reg', action='store_true', help="An optional flag to regularize network")
  parser.add_argument('--save', action='store_true', help="An optional flag to save data")
  parser.add_argument('--all', action='store_true', help="An optional flag to run on all data")
  parser.add_argument('--mean', action='store_true', help="An optional flag to use mean variables")
  args = parser.parse_args()
  
  solver = args.solver
  hls = [int(x) for x in args.hls.split("-") if len(args.hls) > 0]

  N = args.ex
  focus = args.focus
  train_time = args.time
  seed = args.seed
  loss = args.loss
  data = args.data
  bound = args.bound
  batch_size = args.batch
  reg = args.reg

  print(args)

  if loss not in milps:
    raise Exception("MILP model %s not known" % loss)

  if solver not in solvers:
    raise Exception("Solver %s not known" % solver)

  data = load_data(data, N, seed)

  if batch_size == 0:
    batch_size = N
  batches = get_batches(data, batch_size)

  architecture = get_architecture(data, hls)
  print("architecture", architecture)

  print_str = "Architecture: %s. N: %s. Solver: %s. Loss: %s. Bound: %s"
  clear_print(print_str % ("-".join([str(x) for x in architecture]), N, solver, loss, bound))

  get_nn = solvers[solver]
  networks = []
  network_vars = []

  ### RUN BATCHES
  batch_num = 0
  for batch in batches:
    clear_print("Batch: %s. Examples: %s-%s" % (batch_num, batch_num*batch_size, (batch_num+1)*batch_size))
    nn = get_nn(milps[loss], batch, architecture, bound, reg)
    nn.train(train_time*60, focus)

    obj = nn.get_objective()
    print("Objective value: ", obj)

    varMatrices = nn.extract_values()

    train_acc = infer_and_accuracy(nn.data['train_x'], nn.data["train_y"], varMatrices, nn.architecture)
    test_acc = infer_and_accuracy(nn.data['test_x'], nn.data["test_y"], varMatrices, nn.architecture)

    print("Training accuracy: %s " % (train_acc))
    print("Testing accuracy: %s " % (test_acc))

    networks.append(nn)
    network_vars.append(varMatrices)

    batch_num += 1


  ### RUN ALL TOGETHER FROM BATCHES WARM START
  if batch_size > 0 and batch_size != N:
    clear_print("Using warm start from batches")
    bound_matrix = get_bound_matrix(network_vars, bound)
    #bound_matrix = get_alt_bound_matrix(network_vars, bound)

    warm_nn = get_nn(milps[loss], data, architecture, bound)
    #warm_nn.warm_start(get_mean_vars(network_vars))
    warm_nn.update_bounds(bound_matrix)
    warm_nn.train(train_time*60, focus)

    obj = warm_nn.get_objective()
    print("Objective value: ", obj)

    varMatrices = warm_nn.extract_values()

    warm_train_acc = infer_and_accuracy(warm_nn.data['train_x'], warm_nn.data["train_y"], varMatrices, warm_nn.architecture)
    warm_test_acc = infer_and_accuracy(warm_nn.data['test_x'], warm_nn.data["test_y"], varMatrices, warm_nn.architecture)

    print("Training accuracy: %s " % (warm_train_acc))
    print("Testing accuracy: %s " % (warm_test_acc))

    warm_nn_time = sum([i.get_runtime() for i in networks]) + warm_nn.get_runtime()
    clear_print("Warm start run time: %.2f. Accuracy: %s" % (warm_nn_time, warm_test_acc))


  ### RUN ALL DATA AT ONCE
  if args.all:
    clear_print("All data at once")
    nn = get_nn(milps[loss], data, architecture, bound)
    nn.train(train_time*60, focus)

    obj = nn.get_objective()
    print("Objective value: ", obj)

    varMatrices = nn.extract_values()

    train_acc = infer_and_accuracy(nn.data['train_x'], nn.data["train_y"], varMatrices, nn.architecture)
    test_acc = infer_and_accuracy(nn.data['test_x'], nn.data["test_y"], varMatrices, nn.architecture)

    print("Training accuracy: %s " % (train_acc))
    print("Testing accuracy: %s " % (test_acc))


    clear_print("All at once run time: %.2f. Accuracy: %s" % (nn.get_runtime(), test_acc))

  ### TAKE MEAN VARIABLES AND RUN INFERENCE
  if args.mean:
    mean_vars = get_mean_vars(network_vars)

    train_acc = infer_and_accuracy(nn.data['train_x'], nn.data["train_y"], mean_vars, nn.architecture)
    test_acc = infer_and_accuracy(nn.data['test_x'], nn.data["test_y"], mean_vars, nn.architecture)

    clear_print("Training accuracy for mean parameters: %s" % (train_acc))
    clear_print("Testing accuracy for mean parameters: %s" % (test_acc))

  w1 = varMatrices['w_1']
  b1 = varMatrices['b_1']
  act1 = varMatrices['act_1']
  train = nn.data['train_x']

  tmp_inf = np.dot(train, w1) + b1
  tmp_inf[tmp_inf >= 0] = 1
  tmp_inf[tmp_inf < 0] = -1
  inf = np.dot(tmp_inf, varMatrices['w_2']) + varMatrices['b_2']
  norm = 2*inf / ((hls[0]+1)*bound)
  set_trace()

  if args.save:
    if solver == 'gurobi':
      progress = nn.m._progress
    else:
      progress = nn.progress
    DS = DataSaver(nn, architecture, N, focus, solver)
    DS.plot_periodic(progress)
    DS.save_json(train_acc, test_acc)

