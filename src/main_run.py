from milp.cplex_nn import get_cplex_nn
from milp.gurobi_nn import get_gurobi_nn
from milp.min_w import MIN_W
from milp.max_correct import MAX_CORRECT
from milp.min_hinge import MIN_HINGE
from milp.sat_margin import SAT_MARGIN
from milp.max_m import MAX_M
from gd.gd_nn import GD_NN
from helper.misc import inference, infer_and_accuracy, clear_print, get_bound_matrix, get_mean_bound_matrix, \
                        get_mean_vars,get_median_vars, get_network_size,strip_network
from helper.data import load_data, get_batches, get_architecture
from helper.save_data import DataSaver
from helper.fairness import equalized_odds, demographic_parity
import argparse
import numpy as np
import time
from pdb import set_trace

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
  parser.add_argument('--fair', default="", type=str)
  parser.add_argument('--reg', default=0, type=float)
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
  fair = args.fair

  print(args)

  if loss not in milps and loss not in gds:
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

  time_start = time.time()

  ### RUN BATCHES
  batch_num = 0
  for batch in batches:
    clear_print("Batch: %s. Examples: %s-%s" % (batch_num, batch_num*batch_size, (batch_num+1)*batch_size))
    
    if loss in milps:
      nn = get_nn(milps[loss], batch, architecture, bound, reg, fair)
      nn.train(train_time*60, focus)
    else:
      lr = 1e-2
      nn = GD_NN(batch, batch_size, architecture, lr, bound, seed)
      nn.train(train_time*60)

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
  if batch_size > 0 and batch_size != N and not args.mean:
    clear_print("Using warm start from batches")
    bound_matrix = get_bound_matrix(network_vars, bound)
    #bound_matrix = get_alt_bound_matrix(network_vars, bound)

    warm_nn = get_nn(milps[loss], data, architecture, bound, reg, fair)
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
    nn = get_nn(milps[loss], data, architecture, bound, reg, fair)
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
    time_end = time.time()
    total_time = time_end - time_start

    mean_vars = get_mean_vars(network_vars)

    train_acc = infer_and_accuracy(nn.data['train_x'], nn.data["train_y"], mean_vars, nn.architecture)
    test_acc = infer_and_accuracy(nn.data['test_x'], nn.data["test_y"], mean_vars, nn.architecture)

    clear_print("Training accuracy for mean parameters: %s" % (train_acc))
    clear_print("Testing accuracy for mean parameters: %s" % (test_acc))
    clear_print("Time to run all batches: %.2f" % (total_time))
    
    mean_bound_matrix = get_mean_bound_matrix(network_vars, bound)
    mean_warm_nn = get_nn(milps[loss], data, architecture, bound, reg, fair)
    #mean_warm_nn.warm_start(mean_vars)
    mean_warm_nn.update_bounds(mean_bound_matrix)

    mean_warm_nn.train(train_time*60, focus)

    obj = mean_warm_nn.get_objective()
    print("Objective value: ", obj)

    mean_warm_varMatrices = mean_warm_nn.extract_values()

    print("Objective value: ", obj)

    warm_train_acc = infer_and_accuracy(mean_warm_nn.data['train_x'], mean_warm_nn.data["train_y"], mean_warm_varMatrices, mean_warm_nn.architecture)
    warm_test_acc = infer_and_accuracy(mean_warm_nn.data['test_x'], mean_warm_nn.data["test_y"], mean_warm_varMatrices, mean_warm_nn.architecture)

    print("Training accuracy: %s " % (warm_train_acc))
    print("Testing accuracy: %s " % (warm_test_acc))

    warm_nn_time = sum([i.get_runtime() for i in networks]) + mean_warm_nn.get_runtime()
    clear_print("Warm start run time: %.2f. Accuracy: %s" % (warm_nn_time, warm_test_acc))

  w1 = varMatrices['w_1']
  b1 = varMatrices['b_1']
  if len(architecture) > 2:
    w2 = varMatrices['w_2']
    b2 = varMatrices['b_2']
    train = nn.data['train_x']

    tmp_inf = np.dot(train, w1) + b1
    tmp_inf[tmp_inf >= 0] = 1
    tmp_inf[tmp_inf < 0] = -1
    inf = np.dot(tmp_inf, varMatrices['w_2']) + varMatrices['b_2']
    norm = 2*inf / ((hls[0]+1)*bound)

  net_size = get_network_size(architecture, bound)
  print("Network memory: %s Bytes" % net_size)

  stripped,new_arch = strip_network(varMatrices, architecture)
  new_net_size = get_network_size(new_arch, bound)
  if new_net_size != net_size:
    print("New Network memory: %s Bytes" % new_net_size)

    stripped_train_acc = infer_and_accuracy(nn.data['train_x'], nn.data["train_y"], stripped, new_arch)
    stripped_test_acc = infer_and_accuracy(nn.data['test_x'], nn.data["test_y"], stripped, new_arch)

    print("Stripped Training accuracy: %s " % (stripped_train_acc))
    print("Stripped Testing accuracy: %s " % (stripped_test_acc))

  if fair:
    female_train = data['train_x'][:,64]
    male_train = data['train_x'][:,65]
    labels_train = np.array(inference(data['train_x'], varMatrices, architecture))
    female_perc_train = (female_train*labels_train).sum() / labels_train.sum()
    print("female_perc_train", female_perc_train)
    male_perc_train = (male_train*labels_train).sum() / labels_train.sum()
    print("male_perc_train", male_perc_train)

    female_test = data['test_x'][:,64]
    male_test = data['test_x'][:,65]
    labels_test = np.array(inference(data['test_x'], varMatrices, architecture))

  if fair == "EO":

    clear_print("Equalized Odds:")

    tr_p111, tr_p101, tr_p110, tr_p100 = equalized_odds(data['train_x'], labels_train, data['train_y'])
    
    print("train_p111: %.3f" % (tr_p111))
    print("train_p101: %.3f" % (tr_p101))
    print("train_p110: %.3f" % (tr_p110))
    print("train_p100: %.3f" % (tr_p100))
    p111, p101, p110, p100 = equalized_odds(data['test_x'], labels_test, data['test_y'])

    print("test_p111: %.3f" % (p111))
    print("test_p101: %.3f" % (p101))
    print("test_p110: %.3f" % (p110))
    print("test_p100: %.3f" % (p100))

    print("NN p111: %.3f" % (nn.female_pred1_true1.getValue()))
    print("NN p101: %.3f" % (nn.male_pred1_true1.getValue()))
    print("NN p110: %.3f" % (nn.female_pred1_true0.getValue()))
    print("NN p100: %.3f" % (nn.male_pred1_true0.getValue()))

  elif fair == "DP":

    clear_print("Demographic Parity:")

    tr_p11, tr_p10 = demographic_parity(data['train_x'], labels_train, data['train_y'])
    print("train_p11: %.3f" % (tr_p11))
    print("train_p10: %.3f" % (tr_p10))

    p11, p10 = demographic_parity(data['test_x'], labels_test, data['test_y'])
    print("test_p11: %.3f" % (p11))
    print("test_p10: %.3f" % (p10))

    print("NN p11: %.3f" % (nn.female_pred1.getValue()))
    print("NN p10: %.3f" % (nn.male_pred1.getValue()))

  set_trace()

  if args.save:
    if solver == 'gurobi':
      progress = nn.m._progress
    else:
      progress = nn.progress
    DS = DataSaver(nn, architecture, N, focus, solver)
    DS.plot_periodic(progress)
    #DS.save_json(train_acc, test_acc)

