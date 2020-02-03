from milp.cplex_bnn import get_cplex_bnn
from milp.gurobi_bnn import get_gurobi_bnn
from milp.min_w_bnn import MIN_W_BNN
from milp.max_correct_bnn import MAX_CORRECT_BNN
from milp.min_hinge_bnn import MIN_HINGE_BNN
from milp.min_hinge_reg_bnn import MIN_HINGE_REG_BNN
from helper.misc import inference, calc_accuracy
from helper.save_data import DataSaver
from globals import ARCHITECTURES
import argparse

milps = {
  "min_w": MIN_W_BNN,
  "max_correct": MAX_CORRECT_BNN,
  "min_hinge": MIN_HINGE_BNN,
  "min_hinge_reg": MIN_HINGE_REG_BNN
}  

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--solver', default="gurobi", type=str)
  parser.add_argument('--arch', default=2, type=str)
  parser.add_argument('--ex', default=3, type=int)
  parser.add_argument('--focus', default=0, type=int)
  parser.add_argument('--time', default=1, type=float)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--loss', default="min_w", type=str)
  parser.add_argument('--save', action='store_true', help="An optional flag to save data")
  args = parser.parse_args()
  
  solver = args.solver
  if "-" in args.arch:
    architecture = [int(x) for x in args.arch.split("-")]
  else:
    architecture = ARCHITECTURES[int(args.arch)]
  numExamples = args.ex
  focus = args.focus
  time = args.time
  seed = args.seed
  loss = args.loss

  print(args)

  if loss not in milps:
    raise Exception("MILP model not known")

  if solver == 'gurobi':
    bnn = get_gurobi_bnn(milps[loss], numExamples, architecture, seed)
  elif solver =='cplex':
    bnn = get_cplex_bnn(milps[loss], numExamples, architecture, seed)
  else:
    raise Exception("Solver %s not known" % solver)
  bnn.train(time*60, focus)

  obj = bnn.get_objective()
  print("Objective value: ", obj)

  varMatrices = bnn.extract_values()

  infer_train = inference(bnn.data["train_x"], varMatrices, bnn.architecture)
  infer_test = inference(bnn.data["test_x"], varMatrices, bnn.architecture)

  train_acc = calc_accuracy(infer_train, bnn.data["train_y"])
  test_acc = calc_accuracy(infer_test, bnn.data["test_y"])

  print("Training accuracy: %s " % (train_acc))
  print("Testing accuracy: %s " % (test_acc))

  #from pdb import set_trace
  #set_trace()

  if args.save:
    if solver == 'gurobi':
      progress = bnn.m._progress
    else:
      progress = bnn.progress
    DS = DataSaver(bnn, architecture, numExamples, focus, solver)
    DS.plot_periodic(progress)
    DS.save_json(train_acc, test_acc)

