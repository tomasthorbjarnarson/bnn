from milp.icarte_bnn import IcarteBNN
from helper.misc import inference
from helper.plotter import plot_periodic
from datetime import datetime
import numpy as np
import json
import math
from pdb import set_trace

now = datetime.now()
nowStr = now.strftime("%d/%m/%Y %H:%M:%S")

arch1 = [784, 10]
arch2 = [784, 16, 10]
arch3 = [784, 16, 16, 10]

timerun = 120
mipFocus = 3

numExamples = 20
architecture = arch2

bnn = IcarteBNN(numExamples, architecture)

bnn.train(timerun*60, mipFocus)

timeElapsed = math.floor(bnn.m._periodic[-1][3])

archStr = ','.join([str(z) for z in architecture])
runTitle = 'Arch:%s-Exs:%s-Time:%s-MIPFocus:%s' % (archStr,numExamples,timeElapsed,mipFocus)

plot_periodic(bnn.m._periodic, runTitle)

print("Train labels: ")
print(bnn.train_y)
obj = bnn.m.getObjective()
print("Objective value: ", obj.getValue())

varMatrices = bnn.extract_values()


infer_train = inference(bnn.train_x, varMatrices, bnn.architecture)
infer_test = inference(bnn.test_x, varMatrices, bnn.architecture)

train_acc = 0
test_acc = 0

for i, label in enumerate(infer_train):
  if label == bnn.train_y[i]:
    train_acc += 1
train_acc = train_acc/len(bnn.train_y)

for i, label in enumerate(infer_test):
  if label == bnn.test_y[i]:
    test_acc += 1
test_acc = test_acc/len(bnn.test_y)

print("Training accuracy: %s " % (train_acc*100))
print("Testing accuracy: %s " % (test_acc*100))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

save = {
  'datetime': nowStr,
  'architecture': architecture,
  'numExamples': numExamples,
  'obj': bnn.m.ObjVal,
  'bound': bnn.m.ObjBound,
  'gap': bnn.m.MIPGap,
  'nodecount': bnn.m.NodeCount,
  'timerun': timerun,
  'MIPFocus': mipFocus,
  'trainingAcc': train_acc,
  'testingAcc': test_acc,
  'variables': varMatrices
}

with open('results/json/%s.json' % runTitle, 'w') as f:
  json.dump(save, f, cls=NumpyEncoder)