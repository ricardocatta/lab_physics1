import statistical_plot as sp
import statistical_module as sm
import numpy as np
import matplotlib.pylab as plt

x = [1, -1, 4, -2]
y = [2, -1, 3, -3]
x = np.array(x)
y = np.array(y)

m, b = sm.least_square(x, y, 2)
print("m = ", m)
print("b = ", b)


y1 = y
x1 = x
y1 = m * x1 + b

erro_quaratico = sm.squared_error(y, y1, 3)
erro_quaratico_mean = sm.mean_squared_error(y, 3)
erro_regressao = sm.erro_regressao(y, y1, 3)

print("erro quadratico = ", erro_quaratico)
print("erro quadratico medio = ", erro_quaratico_mean)
print("erro da regressão = ", erro_regressao)

plt.plot(x, y, "*")
plt.plot(x1, y1)
plt.grid()
plt.show()