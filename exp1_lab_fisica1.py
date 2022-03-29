from cProfile import label
import statistical_module as sm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Importa os dados coletados do experimento para o relatório 1
#dataframe1 = pd.read_csv("relatorio1.csv", keep_default_na=True) #importa a lista de dados
dataframe1 = pd.read_csv("relatorio11.csv", keep_default_na=True) #importa a lista de dados


# Cada time representa uma coluna de tempos marcados que a bolinha percorreu para uma determinada  distância
time1 = dataframe1.time1
time2 = dataframe1.time2
time3 = dataframe1.time3


# Lista com as posições marcadas 
s1 = dataframe1.position1

# Transformando os times em arrays
t1 = np.array(time1)
t2 = np.array(time2)
t3 = np.array(time3)



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

print("t_mean_1 = ", t_mean_1)

x1 = np.array(t_mean_1)
y = np.array(s1)

# Cálculo dos coeficientes angulares (m1, m2) e dos coeficientes lineares (b1,b2) 
# por meio do método dos mínimos quadrados, que foi implementado no módulo sm.
m1, b1 = sm.least_square(x1, y, 2)


y1 = y
x1 = x1

# Reta linearizada para a altura h1
y1 = m1 * x1 + b1

print("b1 = ", b1)
print("m1 = ", m1)
n1 = m1
k1 = np.exp(b1)


statistical_erro_time1 = sm.statistical_error(t_mean_1, 2)
statistical_erro_position1 = sm.statistical_error(s1, 2)
erro_associado_time1 = sm.erro_associado(t_mean_1, 6, 0.01)
erro_associado_position1 = sm.erro_associado(s1, 6, 0.0005)


print("statistical_erro_time1 = ", statistical_erro_time1)
print("statistical_erro_position1 = ", statistical_erro_position1)
print("erro_associado_time1 = ", erro_associado_time1)
print("erro_associado_position1 = ", erro_associado_position1)






def plot_h1(t_mean_1, s1):
    """
    Plota o gráfico para a altura h1.

    OUTPUT:

    Retorna o gráfico com os erros associados e a reta linearizada pelo MMQ.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=150)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('$\ln{y}$ $[m]$')
    axes1.set_xlabel('$\ln{t}$ $[s]$')
    
    # Derivada da função para a primeira altura
    erro_associado_time1 = sm.erro_associado(t_mean_1, 6, 0.01)

    erro_associado_position1 = sm.erro_associado(s1, 4, 0.0005)

    dydt1 = 1 / (t_mean_1)
    propa_ince_t1 = sm.propaga_incerteza_3D(dydt1, 0, 0, erro_associado_time1, 0, 0)

    dsdt1 = 1 / (s1)
    propa_ince_s1 = sm.propaga_incerteza_3D(dsdt1, 0, 0, erro_associado_position1, 0, 0)


    

    print("propa_ince_t1 = ", np.round(propa_ince_t1, 1))

    print("propa_ince_s1 = ", np.round(propa_ince_s1, 1))
    ln_t1 = np.log(t_mean_1)
    ln_s1 = np.log(s1)
    ln_t1 = np.round(ln_t1, 1)
    ln_s1 = np.round(ln_s1, 1)

    print("ln_t1 = ", np.round(ln_t1, 1))
    print("ln_s1 = ", np.round(ln_s1, 1))
    
    m1, b1 = sm.least_square(ln_t1, ln_s1, 5)
   

    print("\n n = \n", m1)
    
    y1 = m1 * ln_t1 + b1
    incert_n = sm.sigma_m(propa_ince_s1, ln_t1)
    incert_lnk = sm.sigma_b(propa_ince_s1, ln_t1)

    print("\n incerteza de n = \n", incert_n)
    
    k = np.exp(b1)
    print("\n lnk = \n", b1)
    print("\n k = \n", k)

    print("\n incerteza de lnk = \n", incert_lnk)
    print("\n incerteza de k = \n", np.exp(incert_lnk))

    plt.plot(ln_t1, y1, label ="linearização por MMQ")

    #Coloca a barra de erro%
    ls = ''


    for i in range(len(y1)):
        if i == 0:
            plt.errorbar(ln_t1[i], ln_s1[i], xerr=propa_ince_t1[i],  yerr=propa_ince_s1[i], linestyle=ls, marker='o',label ="ponto experimental P1")  
        elif i == 1:
            plt.errorbar(ln_t1[i], ln_s1[i], xerr=propa_ince_t1[i],  yerr=propa_ince_s1[i], linestyle=ls, marker='o',label ="ponto experimental P2")
        elif i == 2:
            plt.errorbar(ln_t1[i], ln_s1[i], xerr=propa_ince_t1[i],  yerr=propa_ince_s1[i], linestyle=ls, marker='o',label ="ponto experimental P3")
        elif i == 3:
            plt.errorbar(ln_t1[i], ln_s1[i], xerr=propa_ince_t1[i],  yerr=propa_ince_s1[i], linestyle=ls, marker='o',label ="ponto experimental P4")
        elif i == 4:
            plt.errorbar(ln_t1[i], ln_s1[i], xerr=propa_ince_t1[i],  yerr=propa_ince_s1[i], linestyle=ls, marker='o',label ="ponto experimental P5")
        else:
            plt.errorbar(ln_t1[i], ln_s1[i], xerr=propa_ince_t1[i],  yerr=propa_ince_s1[i], linestyle=ls, marker='o',label ="ponto experimental P7")  
    fig.tight_layout()
    plt.title("Gráfico da linearização")
    plt.legend(loc='best')
    plt.show()
    return m1, k

def plot_h2(m1, k, t_mean_1):
    """
    Plota o gráfico para a altura h2.

    OUTPUT:

    Retorna o gráfico com os erros associados e a reta linearizada pelo MMQ.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=140)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('y $[m]$')
    axes1.set_xlabel('t $[s]$')
    y2 = k * (t_mean_1 ** (m1))
    plt.plot(t_mean_1, y2, label ="$y = kt^{n}$")
    #Coloca a barra de erro%
    fig.tight_layout()
    plt.title("Posição x tempo")
    plt.legend(loc='best')
    plt.show()

def plot_h1_h2():
    """
    Plota o gráfico para a altura h1 e h2.

    OUTPUT:

    Retorna o gráfico com a linearização e com os ponto experimental para as duas alturas.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('Posição $[m]$')
    axes1.set_xlabel('tempo $[s]$')
    plt.plot(x1, y1, label ="linearização por MMQ para h1")
    plt.plot(x2, y2, label ="linearização por MMQ para h2")
    plt.scatter(x1, y, label ="ponto experimental para h1")
    plt.scatter(x2, y, label ="ponto experimental para h2")
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()


# Descomente a função que se deseja plotar o gráfico

plot_h1(t_mean_1, s1)
#m1, k = plot_h1(t_mean_1, s1)
#plot_h2(m1, k, t_mean_1)