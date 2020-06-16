import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
import seaborn as sns

def get_vars_one_hl(T, N_0, N_L):

  weights = N_0*16 + 16*N_L
  biases = 16 + N_L
  conns = T*16*N_L
  acts = T*16
  outs = T*N_L

  return weights + biases + conns + acts + outs

def get_vars_two_hl(T, N_0, N_L):

  weights = N_0*16 + 16*16 + 16*N_L
  biases = 16 + 16 + N_L
  conns = T*(16*16 + 16*N_L)
  acts = T*(16+16)
  outs = T*N_L

  return weights + biases + conns + acts + outs

def get_constrs_one_hl(T, N_0, N_L):
  conns = 4*T*16*N_L
  acts = 2*T*16
  outs = T*(2*N_L+1)

  return conns+acts+outs

def get_constrs_two_hl(T, N_0, N_L):
  conns = 4*T*(16*16 + 16*N_L)
  acts = 2*T*(16 + 16)
  outs = T*(2*N_L+1)

  return conns+acts+outs


if __name__ == '__main__':
  x = [0,200]

  mnist_vars_one_hl = [get_vars_one_hl(x[0], 784, 10),get_vars_one_hl(x[1], 784, 10)]
  adult_vars_one_hl = [get_vars_one_hl(x[0], 108, 2),get_vars_one_hl(x[1], 108, 2)]
  heart_vars_one_hl = [get_vars_one_hl(x[0], 21, 2),get_vars_one_hl(x[1], 21, 2)]

  print("mnist_vars_one_hl", mnist_vars_one_hl)
  print("adult_vars_one_hl", adult_vars_one_hl)
  print("heart_vars_one_hl", heart_vars_one_hl)

  mnist_vars_two_hl = [get_vars_two_hl(x[0], 784, 10),get_vars_two_hl(x[1], 784, 10)]
  adult_vars_two_hl = [get_vars_two_hl(x[0], 108, 2),get_vars_two_hl(x[1], 108, 2)]
  heart_vars_two_hl = [get_vars_two_hl(x[0], 21, 2),get_vars_two_hl(x[1], 21, 2)]
  print("mnist_vars_two_hl", mnist_vars_two_hl)
  print("adult_vars_two_hl", adult_vars_two_hl)
  print("heart_vars_two_hl", heart_vars_two_hl)

  sns.set_style("darkgrid")

  plt.figure(1, figsize=(6,4))

  plt.plot(x, mnist_vars_one_hl, linestyle="-.", label="MNIST 1 HL", color=(1,0,0,1))
  plt.plot(x, mnist_vars_two_hl, linestyle="-.", label="MNIST 2 HL", color=(1,0,0,0.5))
  
  plt.plot(x, adult_vars_one_hl, linestyle="-", label="Adult 1 HL", color=(0.2,1,0.1,1))
  plt.plot(x, adult_vars_two_hl, linestyle="-", label="Adult 2 HL", color=(0.2,1,0.1,0.5))
  
  plt.plot(x, heart_vars_one_hl, linestyle=":", label="Heart 1 HL", color=(0,0,1,1))
  plt.plot(x, heart_vars_two_hl, linestyle=":", label="Heart 2 HL", color=(0,0,1,0.5))

  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel("Number of variables")
  plt.title("Variables in models for different datasets")

  plt.savefig("model_vars.png", bbox_inches='tight')


  mnist_constrs_one_hl = [get_constrs_one_hl(x[0], 784, 10),get_constrs_one_hl(x[1], 784, 10)]
  adult_constrs_one_hl = [get_constrs_one_hl(x[0], 108, 2),get_constrs_one_hl(x[1], 108, 2)]
  heart_constrs_one_hl = [get_constrs_one_hl(x[0], 21, 2),get_constrs_one_hl(x[1], 21, 2)]

  print("mnist_constrs_one_hl", mnist_constrs_one_hl)
  print("adult_constrs_one_hl", adult_constrs_one_hl)
  print("heart_constrs_one_hl", heart_constrs_one_hl)

  mnist_constrs_two_hl = [get_constrs_two_hl(x[0], 784, 10),get_constrs_two_hl(x[1], 784, 10)]
  adult_constrs_two_hl = [get_constrs_two_hl(x[0], 108, 2),get_constrs_two_hl(x[1], 108, 2)]
  heart_constrs_two_hl = [get_constrs_two_hl(x[0], 21, 2),get_constrs_two_hl(x[1], 21, 2)]
  print("mnist_constrs_two_hl", mnist_constrs_two_hl)
  print("adult_constrs_two_hl", adult_constrs_two_hl)
  print("heart_constrs_two_hl", heart_constrs_two_hl)

  plt.figure(2, figsize=(6,4))
  plt.plot(x, mnist_constrs_one_hl, linestyle="-.", label="MNIST 1 HL", color=(1,0,0,1))
  plt.plot(x, mnist_constrs_two_hl, linestyle="-.", label="MNIST 2 HL", color=(1,0,0,0.5))
  
  plt.plot(x, adult_constrs_one_hl, linestyle="-", label="Adult 1 HL", color=(0.2,1,0.1,1))
  plt.plot(x, adult_constrs_two_hl, linestyle="-", label="Adult 2 HL", color=(0.2,1,0.1,0.5))
  
  plt.plot(x, heart_constrs_one_hl, linestyle=":", label="Heart 1 HL", color=(0,0,1,1))
  plt.plot(x, heart_constrs_two_hl, linestyle=":", label="Heart 2 HL", color=(0,0,1,0.5))

  plt.legend()
  plt.xlabel("Number of examples")
  plt.ylabel("Number of constraints")
  plt.title("Constraints in models for different datasets")
  plt.savefig("model_constrs.png", bbox_inches='tight')

  plt.show()