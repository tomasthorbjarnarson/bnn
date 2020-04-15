import numpy as np
import matplotlib.pyplot as plt


fun = lambda z: np.square(np.maximum(0,0.5-z))
x = list(range(-3,4))
a = np.linspace(-3,3, 100)

vec_fun = np.vectorize(fun)

y = vec_fun(x)
b = vec_fun(a)

plt.plot(x,y, label="Pieceise Linear Squared Hinge Loss", color='r', marker='.', markersize=6)
plt.plot(a,b, label="Squared Hinge Loss", color='b')
plt.legend()
plt.xlabel("x")
plt.ylabel("y", rotation=0)
plt.title("Squared Hinge Loss Representations")
plt.savefig("sq-hinge.png")
plt.show()
