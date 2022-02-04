import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import statistical_module as sm


dataframe = pd.read_csv("lista.csv", keep_default_na=True) #importa a lista de dados

casas = 2 #casas decimais que se deseja arredondar
erro_inst = 0.1
derivada = 6.2
erro = 0.1
mean = sm.mean(dataframe, casas)
std = sm.std(dataframe, casas)
statistical_error = sm.statistical_error(dataframe, casas)
erro_associado = sm.erro_associado(dataframe, casas, erro_inst)
delta_f = sm.propaga_incerteza_1D(dataframe, derivada, erro)
print("O valor médio é: ", mean)
print("O desvio padrão foi: ", std)
print("O statistical_error foi: ", statistical_error)
print("O erro_associado foi: ", erro_associado)
print("O delta_f foi: ", delta_f)





    
