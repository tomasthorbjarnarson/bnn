from milp.bnn import BNN
from helper.misc import inference, calc_accuracy
from helper.save_data import DataSaver
import argparse

ARCHITECTURES = {
  1: [784, 10],
  2: [784, 16, 10],
  3: [784, 16, 16, 10]
}


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--arch', default=2, type=int)
  parser.add_argument('--ex', default=3, type=int)
  parser.add_argument('--focus', default=3, type=int)
  parser.add_argument('--time', default=1, type=float)
  parser.add_argument('--save', action='store_true', help="An optional flag to save data")
  args = parser.parse_args()
  
  architecture = ARCHITECTURES[args.arch]
  numExamples = args.ex
  focus = args.focus
  time = args.time

  print(args)

  bnn = BNN(numExamples, architecture)
  bnn.train(time*60, focus)

  obj = bnn.m.getObjective()
  print("Objective value: ", obj.getValue())

  varMatrices = bnn.extract_values()

  infer_train = inference(bnn.train_x, varMatrices, bnn.architecture)
  infer_test = inference(bnn.test_x, varMatrices, bnn.architecture)

  train_acc = calc_accuracy(infer_train, bnn.train_y)
  test_acc = calc_accuracy(infer_test, bnn.test_y)

  print("Training accuracy: %s " % (train_acc))
  print("Testing accuracy: %s " % (test_acc))

  if args.save:
    DS = DataSaver(bnn, architecture, numExamples, focus, time)
    DS.plot_periodic()
    DS.save_json(train_acc, test_acc)

