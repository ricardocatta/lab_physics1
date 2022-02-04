import statistical_plot as sp
import statistical_module as sm
import numpy as np
import matplotlib.pylab as plt


################################################################
# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com
# Main
# Interface para plotar os gráficos dos experimentos
################################################################


######################################################################################
# Para plotar os resultados do experimento 1:
######################################################################################

V1 = sp.dataframe1.V
I1 = sp.dataframe1.I

#sp.exp1(V1, I1, sp.dataframe1)

######################################################################################
# Para plotar os resultados do experimento 2:
######################################################################################

V2 = sp.dataframe2.V
I2 = sp.dataframe2.I

#sp.exp2(V2, I2, sp.dataframe2)

######################################################################################
# Para plotar os resultados do experimento 3:
######################################################################################

V3 = sp.dataframe3.V
I3 = sp.dataframe3.I

#sp.exp3(V3, I3, sp.dataframe3)

######################################################################################################
# Para plotar os resultados da Intensidade em função do comprimento de onda dos experimentos 1, 2 e 3:
######################################################################################################

I_data_azul = sp.I_data_azul
lamb_data_azul = sp.lamb_data_azul
I_data_verde = sp.I_data_verde
lamb_data_verde = sp.lamb_data_verde
I_data_vermelho = sp.I_data_vermelho
lamb_data_vermelho = sp.lamb_data_vermelho
#sp.intensidade(I_data_azul, lamb_data_azul, I_data_verde, lamb_data_verde, I_data_vermelho, lamb_data_vermelho)

######################################################################################################
# Calculo da constante de Plank e sua propagação de incerteza (LED verde)
######################################################################################################

lambda_verde = 500 * 10 ** (-9)
V_min_verde = 2.6

def planck_constant_verde(lambda_verde, V_min_verde):
    carga = 1.602176634*10**(-19) # Em módulo
    c_luz = 3 * 10 ** (8)
    freq = c_luz / lambda_verde
    h_planck = carga * V_min_verde / freq
    h_planck_literatura = 6.626 * 10 ** (-34)
    print("h_planck = ", h_planck)
    erro_relativo = abs(h_planck- h_planck_literatura) / h_planck_literatura
    print("erro_relativo = ", np.round(erro_relativo,3))
    carga = -1.602176634*10**(-19)
    c_luz = 3 * 10 ** (8)
    erro_lambda = (0.001 * 10 ** (-9))
    freq = c_luz / lambda_verde
    delta_freq = (c_luz / (lambda_verde **2)) * erro_lambda
    derivada_h_f = -carga * V_min_verde / (freq **2)
    delta_V = 0.1
    derivada_h_V = carga / freq
    erro_propagado_planck = sm.propaga_incerteza_2D(derivada_h_f, derivada_h_V, delta_freq, delta_V)
    print("erro_propagado_planck = ", erro_propagado_planck)
    return h_planck

########################################################################
# Calculo da constante de Plank e sua propagação de incerteza (LED azul)
########################################################################

lambda_azul = 480 * 10 ** (-9)
V_min_azul = 2.5

def planck_constant_azul(lambda_azul, V_min_azul):
    carga = 1.602176634*10**(-19) # Em módulo
    c_luz = 3 * 10 ** (8)
    freq = c_luz / lambda_azul
    h_planck = carga * V_min_azul / freq
    h_planck_literatura = 6.626 * 10 ** (-34)
    print("h_planck = ", h_planck)
    erro_relativo = abs(h_planck- h_planck_literatura) / h_planck_literatura
    print("erro_relativo = ", np.round(erro_relativo,3))
    carga = -1.602176634*10**(-19)
    c_luz = 3 * 10 ** (8)
    erro_lambda = (0.001 * 10 ** (-9))
    freq = c_luz / lambda_azul
    delta_freq = (c_luz / (lambda_azul **2)) * erro_lambda
    derivada_h_f = -carga * V_min_azul / (freq **2)
    delta_V = 0.1
    derivada_h_V = carga / freq
    erro_propagado_planck = sm.propaga_incerteza_2D(derivada_h_f, derivada_h_V, delta_freq, delta_V)
    print("erro_propagado_planck = ", erro_propagado_planck)
    return h_planck

########################################################################
# Calculo da constante de Plank e sua propagação de incerteza (LED vermelho)
########################################################################

lambda_vermelho = 640 * 10 ** (-9)
V_min_vermelho = 2.0

def planck_constant_vermelho(lambda_vermelho, V_min_vermelho):
    carga = 1.602176634*10**(-19) # Em módulo
    c_luz = 3 * 10 ** (8)
    freq = c_luz / lambda_vermelho
    h_planck = carga * V_min_vermelho / freq
    h_planck_literatura = 6.626 * 10 ** (-34)
    print("h_planck = ", h_planck)
    erro_relativo = abs(h_planck- h_planck_literatura) / h_planck_literatura
    print("erro_relativo = ", np.round(erro_relativo,3))
    carga = -1.602176634*10**(-19)
    c_luz = 3 * 10 ** (8)
    erro_lambda = (0.001 * 10 ** (-9))
    freq = c_luz / lambda_vermelho
    delta_freq = (c_luz / (lambda_vermelho **2)) * erro_lambda
    derivada_h_f = -carga * V_min_vermelho / (freq **2)
    delta_V = 0.1
    derivada_h_V = carga / freq
    erro_propagado_planck = sm.propaga_incerteza_2D(derivada_h_f, derivada_h_V, delta_freq, delta_V)
    print("erro_propagado_planck = ", erro_propagado_planck)
    return h_planck

#planck_constant_verde(lambda_verde, V_min_verde)
#planck_constant_azul(lambda_azul, V_min_azul)
#planck_constant_vermelho(lambda_vermelho, V_min_vermelho)
