from cProfile import label
import statistical_module as sm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Importa os dados coletados do experimento para o relatório 1
dataframe1 = pd.read_csv("relatorio1.csv", keep_default_na=True) #importa a lista de dados

# Cada time representa uma coluna de tempos marcados que a bolinha percorreu para uma determinada  distância
time1 = dataframe1.time1
time2 = dataframe1.time2
time3 = dataframe1.time3
time4 = dataframe1.time4
time5 = dataframe1.time5
time6 = dataframe1.time6

# Lista com as posições marcadas 
s1 = dataframe1.position1

# Transformando os times em arrays
t1 = np.array(time1)
t2 = np.array(time2)
t3 = np.array(time3)
t4 = np.array(time4)
t5 = np.array(time5)
t6 = np.array(time6)


def sum_time(t1, t2, t3):
    """
    Retorna a média dos tempos para a altura considerada.
    
    INPUT:

    - t1 = lista do primeiro tempo medido para a altura considerada;
    - t2 = lista do segundo tempo medido para a altura considerada;
    - t3 = lista do terceiro tempo medido para a altura considerada;

    OUTPUT:

    Retorna a média dos tempos para a altura considerada.

    """
    t_mean = np.zeros(len(t1))
    for i in range(len(t1)):
        t_mean[i] = (t1[i] + t2[i] + t3[i]) / 3
    return t_mean

# tempo médio para a altura h1
t_mean_1 = sum_time(t1, t2, t3)

# tempo médio para a altura h2
t_mean_2 = sum_time(t4, t5, t6)

print("t_mean_1 = ", t_mean_1)
print("t_mean_2 = ", t_mean_2)

x1 = np.array(t_mean_1)
x2 = np.array(t_mean_2)
y = np.array(s1)

# Cálculo dos coeficientes angulares (m1, m2) e dos coeficientes lineares (b1,b2) 
# por meio do método dos mínimos quadrados, que foi implementado no módulo sm.
m1, b1 = sm.least_square(x1, y, 2)
m2, b2 = sm.least_square(x2, y, 2)


y1 = y
x1 = x1

# Reta linearizada para a altura h1
y1 = m1 * x1 + b1

x2 = x2

# Reta linearizada para a altura h2
y2 = m2 * x2 + b2

print("b1 = ", b1)
print("m1 = ", m1)
n1 = m1
k1 = np.exp(b1)

n2 = m2
k2 = np.exp(b2)
print("b2 = ", b2)
print("m2 = ", m2)

#Cálculo das quantidades estatísticas que foram implementadas no módulo sm.
erro_quaratico1 = sm.squared_error(y, y1, 3)
erro_quaratico_mean1 = sm.mean_squared_error(y, 3)
erro_regressao1 = sm.erro_regressao(y, y1, 3)

erro_quaratico2 = sm.squared_error(y, y2, 3)
erro_quaratico_mean2 = sm.mean_squared_error(y, 3)
erro_regressao2 = sm.erro_regressao(y, y2, 3)


print("erro quadratico1 = ", erro_quaratico1)
print("erro quadratico medio1 = ", erro_quaratico_mean1)
print("erro da regressão1 = ", erro_regressao1)

print("erro quadratico2 = ", erro_quaratico2)
print("erro quadratico medio2 = ", erro_quaratico_mean2)
print("erro da regressão2 = ", erro_regressao2)

statistical_erro_time1 = sm.statistical_error(t_mean_1, 2)
statistical_erro_position1 = sm.statistical_error(s1, 2)
erro_associado_time1 = sm.erro_associado(t_mean_1, 2, 0.01)
erro_associado_position1 = sm.erro_associado(s1, 2, 0.0005)

statistical_erro_time2 = sm.statistical_error(t_mean_2, 2)
statistical_erro_position2 = sm.statistical_error(s1, 2)
erro_associado_time2 = sm.erro_associado(t_mean_2, 2, 0.01)
erro_associado_position2 = sm.erro_associado(s1, 2, 0.0005)

print("statistical_erro_time1 = ", statistical_erro_time1)
print("statistical_erro_position1 = ", statistical_erro_position1)
print("erro_associado_time1 = ", erro_associado_time1)
print("erro_associado_position1 = ", erro_associado_position1)

print("statistical_erro_time2 = ", statistical_erro_time2)
print("statistical_erro_position2 = ", statistical_erro_position2)
print("erro_associado_time2 = ", erro_associado_time2)
print("erro_associado_position2 = ", erro_associado_position2)

# Derivada da função para a primeira altura
dydt1 = k1 * n1 * (t_mean_1[1:] ** (n1-1))
propa_ince_h1 = sm.propaga_incerteza_3D(dydt1, 0, 0, 0.01, 0, 0)
print("propa_ince_h1 = ", np.round(propa_ince_h1, 3))

# Derivada da função para a segunda altura
dydt2 = k2 * n2 * (t_mean_2[1:] ** (n2-1))
propa_ince_h2 = sm.propaga_incerteza_3D(dydt2, 0, 0, 0.01, 0, 0)
print("propa_ince_h2 = ", np.round(propa_ince_h2, 3))

def plot_h1():
    """
    Plota o gráfico para a altura h1.

    OUTPUT:

    Retorna o gráfico com os erros associados e a reta linearizada pelo MMQ.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('${y}$ $[m]$')
    axes1.set_xlabel('${t}$ $[s]$')
    plt.plot(x1, y1, label ="linearização por MMQ para h1")
    #Coloca a barra de erro%
    ls = ''
    plt.errorbar(x1, y, xerr=0.77,yerr=0.26,linestyle=ls, marker='o',label ="pontos experimentais de h1")
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()

def plot_h2():
    """
    Plota o gráfico para a altura h2.

    OUTPUT:

    Retorna o gráfico com os erros associados e a reta linearizada pelo MMQ.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('${y}$ $[m]$')
    axes1.set_xlabel('${t}$ $[s]$')
    plt.plot(x2, y2, label ="linearização por MMQ para h2")
    #Coloca a barra de erro%
    ls = ''
    plt.errorbar(x2, y, xerr=0.48,yerr=0.26,linestyle=ls, marker='o',label ="pontos experimentais de h2")
    #plt.legend(loc='best')
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()

def plot_h1_h2():
    """
    Plota o gráfico para a altura h1 e h2.

    OUTPUT:

    Retorna o gráfico com a linearização e com os pontos experimentais para as duas alturas.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('${y}$ $[m]$')
    axes1.set_xlabel('${t}$ $[s]$')
    plt.plot(x1, y1, label ="linearização por MMQ para h1")
    plt.plot(x2, y2, label ="linearização por MMQ para h2")
    plt.scatter(x1, y, label ="pontos experimentais para h1")
    plt.scatter(x2, y, label ="pontos experimentais para h2")
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()


# Descomente a função que se deseja plotar o gráfico

#plot_h1()
#plot_h2()
plot_h1_h2()