from cProfile import label
from re import T
import statistical_module as sm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Importa os dados coletados do experimento para o relatório 1
dataframe1 = pd.read_csv("relatorio_3.csv", keep_default_na=True) #importa a lista de dados

# Cada time representa uma coluna de tempos marcados que a bolinha percorreu para uma determinada  distância
time1 = dataframe1.time1
time2 = dataframe1.time2
time3 = dataframe1.time3
time4 = dataframe1.time4
time5 = dataframe1.time5

# Lista com as posições marcadas 
s1 = dataframe1.position1
# Transformando os times em arrays
t1 = np.array(time1)
t2 = np.array(time2)
t3 = np.array(time3)
t4 = np.array(time4)
t5 = np.array(time5)

t1_mean = sm.mean(t1, 3)
t2_mean = sm.mean(t2, 3)
t3_mean = sm.mean(t3, 3)
t4_mean = sm.mean(t4, 3)
t5_mean = sm.mean(t5, 3)

t_total = [t1_mean, t2_mean, t3_mean, t4_mean, t5_mean]
t_total = np.array(t_total)
erro_instrumental = 0.01
t_ea = [sm.erro_associado(t1, 3, erro_instrumental), sm.erro_associado(t2, 3, erro_instrumental), 
sm.erro_associado(t3, 3, erro_instrumental), sm.erro_associado(t4, 3, erro_instrumental), 
sm.erro_associado(t5, 3, erro_instrumental)]

t_ea = np.array(t_ea)
erro_in_posi = (0.5/1000)
s_ea = sm.erro_associado(s1, 3, erro_in_posi)
print("t_total = ", t_total)
print("Erro associado do tempo = ", np.round(t_ea,2))
print("Erro associado da posição = ", np.round(s_ea,1))



propaga_lnt = sm.propaga_incerteza_3D((1/t_total), 0, 0, t_ea, 0, 0)

print("Propagação de incerteza do lnt = ", np.round(propaga_lnt,2))


propaga_lns = sm.propaga_incerteza_3D((1/s1), 0, 0, s_ea, 0, 0)

propaga_lns = np.array(propaga_lns)
print("Propagação de incerteza do lns = ", np.round(propaga_lns,1))


ln_s = np.log(s1)
ln_t = np.log(t_total)


# Cálculo dos coeficientes angulares (m1, m2) e dos coeficientes lineares (b1,b2) 
# por meio do método dos mínimos quadrados, que foi implementado no módulo sm.
m1, b1 = sm.least_square(ln_t, ln_s, 2)


y1 = ln_s
x1 = ln_t

# Reta linearizada para a altura h1
y1 = m1 * x1 + b1


def plot_lns_lnt(m, b, x, y, delx, dely):
    """
    Plota o gráfico para a altura h1.

    OUTPUT:

    Retorna o gráfico com os erros associados e a reta linearizada pelo MMQ.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('$\ln{s}$ $[m]$')
    axes1.set_xlabel('$\ln{t}$ $[s]$')
    y = b + m * x
    y = np.array(y)
    print("b = ", b)
    print("m = ", m)
    print("ln_s = ", y)
    print("ln_t = ", x)
    plt.plot(x, y, '-', label ="linearização por MMQ")
    #Coloca a barra de erro%
    ls = ''
    for i in range(len(y)):
        if i == 0:
            plt.errorbar(x[i], y[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P1")  
        elif i == 1:
            plt.errorbar(x[i], y[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P2")
        elif i == 2:
            plt.errorbar(x[i], y[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P3")
        elif i == 3:
            plt.errorbar(x[i], y[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P4")
        else:
            plt.errorbar(x[i], y[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P5")  
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()



def plot_s_t(t, s, delt, dels):
    """
    Plota o gráfico para a altura h1.

    OUTPUT:

    Retorna o gráfico com os erros associados e a reta linearizada pelo MMQ.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('${s}$ $[m]$')
    axes1.set_xlabel('${t}$ $[s]$')
    print("s = ", s)
    print("t = ", t)
    plt.scatter(t, s)
    #Coloca a barra de erro%
    ls = ''
    for i in range(len(s)):
        if i == 0:
            plt.errorbar(t[i], s[i], xerr=delt[i],  yerr=dels, linestyle=ls, marker='o',label ="ponto experimental P1")  
        elif i == 1:
            plt.errorbar(t[i], s[i], xerr=delt[i],  yerr=dels, linestyle=ls, marker='o',label ="ponto experimental P2")
        elif i == 2:
            plt.errorbar(t[i], s[i], xerr=delt[i],  yerr=dels, linestyle=ls, marker='o',label ="ponto experimental P3")
        elif i == 3:
            plt.errorbar(t[i], s[i], xerr=delt[i],  yerr=dels, linestyle=ls, marker='o',label ="ponto experimental P4")
        else:
            plt.errorbar(t[i], s[i], xerr=delt[i],  yerr=dels, linestyle=ls, marker='o',label ="ponto experimental P5") 
    
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()
# Descomente a função que se deseja plotar o gráfico

plot_s_t(t_total, s1, t_ea, s_ea)
plot_lns_lnt(m1, b1, x1, y1, propaga_lnt, propaga_lns)