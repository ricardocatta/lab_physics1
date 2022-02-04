import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import statistical_module as sm
import pwlf # plota as funções pelo método piecewise

################################################################
# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com
################################################################

### Leitura dos dados em formato csv
dataframe1 = pd.read_csv("exp1.csv", keep_default_na=True) #importa a lista de dados
dataframe2 = pd.read_csv("exp2.csv", keep_default_na=True) #importa a lista de dados
dataframe3 = pd.read_csv("exp3.csv", keep_default_na=True) #importa a lista de dados

### Leitura dos dados em formato txt
data_azul = np.loadtxt("led-azul_Subt2__0__1.txt")
data_verde = np.loadtxt("led-verde_Subt2__0__2.txt")
data_vermelho = np.loadtxt("led-vermelho_Subt2__0__3.txt")

### Leitura das Intensidades e dos comprimentos de onda

I_data_azul = data_azul[0:,0]
lamb_data_azul = data_azul[0:,1]

I_data_verde = data_verde[0:,0]
lamb_data_verde = data_verde[0:,1]

I_data_vermelho = data_vermelho[0:,0]
lamb_data_vermelho = data_vermelho[0:,1]

### Leitura das Voltagens e correntes dos experimentos 1, 2 e 3

V1 = dataframe1.V
I1 = dataframe1.I 

V2 = dataframe2.V
I2 = dataframe2.I

V3 = dataframe3.V
I3 = dataframe3.I

def intensidade(I_data_azul, lamb_data_azul, I_data_verde, lamb_data_verde, I_data_vermelho, lamb_data_vermelho):
    """
    Plota os gráficos das Intensidades em função do comprimento de onda dos experimentos 1, 2 e 3

    INPUT:

    - I_data_azul = Intensidade do LED azul;
    - lamb_data_azul = Comprimento de onda do LED azul;
    - I_data_verde = Intensidade do LED verde;
    - lamb_data_verde = Comprimento de onda do LED verde;
    - I_data_vermelho = Intensidade do LED vermelho;
    - lamb_data_vermelho = Comprimento de onda do LED vermelho.

    OUTPUT:

    Retorna os gráficos das Intensidades em função do comprimento de onda dos experimentos 1, 2 e 3
    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('I')
    axes1.set_xlabel('$\lambda$')
 
    plt.plot(I_data_azul, lamb_data_azul, 'b',label="LED azul")
    plt.plot(I_data_verde, lamb_data_verde, 'g',label="LED verde")
    plt.plot(I_data_vermelho, lamb_data_vermelho, 'r',label="LED vermelho")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

def exp1(V1, I1, dataframe1):
    """
    Plota os gráficos dos experimento 1

    INPUT:

    - V1 = Dados coletados da voltagem;
    - I1 = Dados coletados da corrente;
    - dataframe1 = Conjunto de dados.

    OUTPUT:

    Retorna os gráficos com as barras de erro e o método piecewise.
    """
    def statistical_V(dataframe1):
        ###########################################################################
        # Calcula as propriedades estatísticas da Voltagem para colocar no grático
        ###########################################################################
        mean_V = sm.mean(dataframe1.V, 2)
        std_V = sm.std(dataframe1.V, 2)
        erro_estatistico_V = sm.statistical_error(dataframe1.V, 2)
        erro_associado_V = sm.erro_associado(dataframe1.V, 2, 0.1)
        alpha = 1
        derivada_V = alpha / dataframe1.I
        propag_incert_V = sm.propaga_incerteza_2D(derivada_V, 0, 0.1, 0)
        print("mean_V = ", mean_V)
        print("std_V = ", std_V)
        print("erro_estatistico_V = ", erro_estatistico_V)
        print("erro_associado_V = ", erro_associado_V)
        print("propag_incert_V = ", propag_incert_V)
    
    def statistical_I(dataframe1):
        ###########################################################################
        # Calcula as propriedades estatísticas da Corrente para colocar no grático
        ###########################################################################
        mean_I = sm.mean(dataframe1.I, 2)
        std_I = sm.std(dataframe1.I, 2)
        erro_estatistico_I = sm.statistical_error(dataframe1.I, 2)
        erro_associado_I = sm.erro_associado(dataframe1.I, 2, 0.1)
        alpha = 1
        beta = 1 / alpha
        derivada_I = beta * np.exp(beta * dataframe1.V)
        propag_incert_I = sm.propaga_incerteza_2D(derivada_I, 0, 0.1, 0)
        print("mean_I = ", mean_I)
        print("std_I = ", std_I)
        print("erro_estatistico_I = ", erro_estatistico_I)
        print("erro_associado_I = ", erro_associado_I)
        print("propag_incert_I = ", propag_incert_I)
    
    statistical_V(dataframe1)
    statistical_I(dataframe1)
    
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('I')
    axes1.set_xlabel('V')

    x = np.array(V1)
    y = np.array(I1)
   
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(2)
    print(breaks)
    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = my_pwlf.predict(x_hat)

    #plt.plot(x, y, 'o')
    plt.scatter(x, y, color="red", label="exp1")
    plt.plot(x_hat, y_hat, '-',color="green", label="método piecewise")
    
    #Coloca a barra de erro%
    ls = ''
    plt.errorbar(V1, I1, xerr=0.19,yerr=0.73, linestyle=ls, marker='o')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

def exp2(V2, I2, dataframe2):
    """
    Plota os gráficos dos experimento 2

    INPUT:

    - V2 = Dados coletados da voltagem;
    - I2 = Dados coletados da corrente;
    - dataframe2 = Conjunto de dados.

    OUTPUT:

    Retorna os gráficos com as barras de erro e o método piecewise.
    """
    def statistical_V(dataframe2):
        ###########################################################################
        # Calcula as propriedades estatísticas da Voltagem para colocar no grático
        ###########################################################################
        mean_V = sm.mean(dataframe2.V, 2)
        std_V = sm.std(dataframe2.V, 2)
        erro_estatistico_V = sm.statistical_error(dataframe2.V, 2)
        erro_associado_V = sm.erro_associado(dataframe2.V, 2, 0.1)
        alpha = 1
        derivada_V = alpha / dataframe2.I
        propag_incert_V = sm.propaga_incerteza_2D(derivada_V, 0, 0.1, 0)
        print("mean_V = ", mean_V)
        print("std_V = ", std_V)
        print("erro_estatistico_V = ", erro_estatistico_V)
        print("erro_associado_V = ", erro_associado_V)
        print("propag_incert_V = ", propag_incert_V)
    
    def statistical_I(dataframe2):
        ###########################################################################
        # Calcula as propriedades estatísticas da Corrente para colocar no grático
        ###########################################################################
        mean_I = sm.mean(dataframe2.I, 2)
        std_I = sm.std(dataframe2.I, 2)
        erro_estatistico_I = sm.statistical_error(dataframe2.I, 2)
        erro_associado_I = sm.erro_associado(dataframe2.I, 2, 0.1)
        alpha = 1
        beta = 1 / alpha
        derivada_I = beta * np.exp(beta * dataframe2.V)
        propag_incert_I = sm.propaga_incerteza_2D(derivada_I, 0, 0.1, 0)
        print("mean_I = ", mean_I)
        print("std_I = ", std_I)
        print("erro_estatistico_I = ", erro_estatistico_I)
        print("erro_associado_I = ", erro_associado_I)
        print("propag_incert_I = ", propag_incert_I)
    
    statistical_V(dataframe2)
    statistical_I(dataframe2)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('I')
    axes1.set_xlabel('V')

    x = np.array(V2)
    y = np.array(I2)
   
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(2)
    print(breaks)
    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = my_pwlf.predict(x_hat)

    #plt.plot(x, y, 'o')
    plt.scatter(x, y, color="red", label="exp2")
    plt.plot(x_hat, y_hat, '-',color="green", label="método piecewise")
    
    #Coloca a barra de erro%
    ls = ''
    plt.errorbar(V2, I2, xerr=0.19,yerr=1.89, linestyle=ls, marker='o')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

def exp3(V3, I3, dataframe3):
    """
    Plota os gráficos dos experimento 3

    INPUT:

    - V3 = Dados coletados da voltagem;
    - I3 = Dados coletados da corrente;
    - dataframe3 = Conjunto de dados.

    OUTPUT:

    Retorna os gráficos com as barras de erro e o método piecewise.
    """
    def statistical_V(dataframe3):
        ###########################################################################
        # Calcula as propriedades estatísticas da Voltagem para colocar no grático
        ###########################################################################
        mean_V = sm.mean(dataframe3.V, 2)
        std_V = sm.std(dataframe3.V, 2)
        erro_estatistico_V = sm.statistical_error(dataframe3.V, 2)
        erro_associado_V = sm.erro_associado(dataframe3.V, 2, 0.1)
        alpha = 1
        derivada_V = alpha / dataframe3.I
        propag_incert_V = sm.propaga_incerteza_2D(derivada_V, 0, 0.1, 0)
        print("mean_V = ", mean_V)
        print("std_V = ", std_V)
        print("erro_estatistico_V = ", erro_estatistico_V)
        print("erro_associado_V = ", erro_associado_V)
        print("propag_incert_V = ", propag_incert_V)
    
    def statistical_I(dataframe3):
        ###########################################################################
        # Calcula as propriedades estatísticas da Corrente para colocar no grático
        ###########################################################################
        mean_I = sm.mean(dataframe3.I, 2)
        std_I = sm.std(dataframe3.I, 2)
        erro_estatistico_I = sm.statistical_error(dataframe3.I, 2)
        erro_associado_I = sm.erro_associado(dataframe3.I, 2, 0.1)
        alpha = 1
        beta = 1 / alpha
        derivada_I = beta * np.exp(beta * dataframe3.V)
        propag_incert_I = sm.propaga_incerteza_2D(derivada_I, 0, 0.1, 0)
        print("mean_I = ", mean_I)
        print("std_I = ", std_I)
        print("erro_estatistico_I = ", erro_estatistico_I)
        print("erro_associado_I = ", erro_associado_I)
        print("propag_incert_I = ", propag_incert_I)
    
    statistical_V(dataframe3)
    statistical_I(dataframe3)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('I')
    axes1.set_xlabel('V')

    x = np.array(V3)
    y = np.array(I3)
   
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(2)
    print(breaks)
    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = my_pwlf.predict(x_hat)

    #plt.plot(x, y, 'o')
    plt.scatter(x, y, color="red", label="exp3")
    plt.plot(x_hat, y_hat, '-',color="green", label="método piecewise")
    
    #Coloca a barra de erro%
    ls = ''
    plt.errorbar(V3, I3, xerr=0.19,yerr=3.91, linestyle=ls, marker='o')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()
    



    
