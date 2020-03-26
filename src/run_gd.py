import tensorflow as tf
import numpy as np

from gd.gd_nn import GD_NN
from helper.data import load_data, get_architecture
from helper.misc import infer_and_accuracy, clear_print

tf.logging.set_verbosity(tf.logging.ERROR)


from pdb import set_trace

random_seed = 31567478618
tf.set_random_seed(random_seed)
#seed = 5369
seed = random_seed
N = 1000
data = load_data("adult", N, seed)
#data = load_data("mnist", N, seed)

hls = [16]

architecture = get_architecture(data, hls)
lr = 1e-3
bound = 0.5
time = 1

print_str = "Architecture: %s. N: %s. LR: %s. Bound: %s"
clear_print(print_str % ("-".join([str(x) for x in architecture]), N, lr, bound))


#nn = BNN(data, N, architecture, lr, seed)
nn = GD_NN(data, N, architecture, lr, bound, seed)
nn.train(max_time=time*60)
nn_y_pred = nn.y_pred.eval(session=nn.sess, feed_dict={nn.x: data['train_x']})
#nn_loss = nn.loss.eval(session=nn.sess, feed_dict={nn.x: nn.X_train, nn.y: nn.oh_y_train})
nn_loss = nn.get_objective()
print("nn_loss", nn_loss)
nn_runtime = nn.get_runtime()
print("nn_runtime", nn_runtime)
varMatrices = nn.extract_values()

train_acc = infer_and_accuracy(data['train_x'], data['train_y'], varMatrices, architecture)
test_acc = infer_and_accuracy(data['test_x'], data['test_y'], varMatrices, architecture)
print("train_acc", train_acc)
print("test_acc", test_acc)


loss = np.square(np.maximum(0, 0.5 - nn_y_pred*data['oh_train_y'])).sum()
print("loss", loss)

w1 = varMatrices['w_1']
b1 = varMatrices['b_1']
w2 = varMatrices['w_2']
b2 = varMatrices['b_2']

x = data['test_x']
y = data['test_y']
foo = np.dot(x, w1) + b1
bar = 1/(1+np.exp(-foo))
tmp = np.dot(bar, w2) + b2
acc = np.equal(np.argmax(tmp, 1), y).sum()/len(y)
set_trace()
