from cProfile import label
import statistical_module as sm
import numpy as np
import pandas as pd

import matplotlib.pylab as plt

dataframe1 = pd.read_csv("relatorio1.csv", keep_default_na=True) #importa a lista de dados

time1 = dataframe1.time1
time2 = dataframe1.time2
time3 = dataframe1.time3
time4 = dataframe1.time4
time5 = dataframe1.time5

s1 = dataframe1.position1

t1 = np.array(time1)
t2 = np.array(time2)
t3 = np.array(time3)
t4 = np.array(time4)
t5 = np.array(time5)


def sum_time(t1, t2, t3, t4, t5):
    t_mean = np.zeros(len(t1))
    for i in range(len(t1)):
        t_mean[i] = (t1[i] + t2[i] + t3[i] + t4[i] + t5[i]) / 5
    return t_mean

t_sum = t1[1] + t2[1] + t3[1] + t4[1] + t5[1]

t_mean = sum_time(t1, t2, t3, t4, t5)
print("t_mean = ", t_mean)

x = np.array(t_mean)
y = np.array(s1)

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



plt.style.use('ggplot')
fig = plt.figure(dpi=110)
axes1 = fig.add_subplot(1, 1, 1)
axes1.set_ylabel('posição $[m]$')
axes1.set_xlabel('tempo $[s]$')
plt.plot(x, y, "*", label ="pontos experimentais")
plt.plot(x1, y1, label ="linearização por MMQ")
plt.legend(loc='best')
fig.tight_layout()
plt.show()