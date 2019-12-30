import matplotlib.pyplot as plt

def plot_periodic(periodic, title):
  per_filtered = [z for z in periodic if z[4] < 0.8]
  x = [z[3] for z in per_filtered]
  y = [z[1] for z in per_filtered]
  y2 = [z[2] for z in per_filtered]

  plt.plot(x,y, label="Best objective")
  plt.plot(x,y2, label="Best bound")
  plt.legend()
  plt.xlabel("Time [s]")
  plt.ylabel("Sum of absolute weights")
  plt.title(title)
  plt.show()