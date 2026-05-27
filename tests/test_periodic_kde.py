import numpy as np
from cantuccio.kde import kde_1d, kde_2d
import matplotlib.pyplot as plt

data = np.random.uniform(0, 10, 1000)

x, pdf = kde_1d(data, bw="scott", n=100, fast=True, periodic=(0, 10))

print("1d Integral:", np.trapezoid(pdf, x))


x2 = np.random.uniform(0, 10, 1000)
y2 = np.random.uniform(0, 10, 1000)

xx, yy, zz = kde_2d(x2, y2, bw="scott", n=50, fast=True, periodic_x=(0, 10), periodic_y=(0, 10))
print("2d min/max shapes:", zz.shape, xx.min(), xx.max())

