import numpy as np
import matplotlib.pylab as plt
import pandas as pd

def mean(dataframe, casas):
    x = dataframe.x_i
    x_mean = np.mean(x)
    return np.round(x_mean, casas)
    
def std(dataframe, casas):
    x = dataframe.x_i
    N = len(x)
    x_mean = np.mean(x)
    x_mean = np.round(x_mean, casas)
    numerador = np.sum((x - x_mean)**2)
    denominador = N - 1
    desvio_p = np.sqrt(numerador/denominador)
    return np.round(desvio_p, casas)

def statistical_error(dataframe, casas):
    x = dataframe.x_i
    N = len(x)
    desvio_p = std(dataframe, casas)
    statisticalerror = desvio_p / np.sqrt(N)
    return np.round(statisticalerror, casas)

def erro_associado(dataframe, casas, erro_inst):
    erro_estatistico = statistical_error(dataframe, casas)
    erro_instrumental = erro_inst
    erroassociado = np.sqrt((erro_estatistico ** 2) + (erro_instrumental ** 2))
    return np.round(erroassociado, casas)

def propaga_incerteza_1D(dataframe, derivada, erro):
    x = dataframe.x_i
    delta_x = erro
    df = derivada
    delta_f = df * delta_x
    return np.round(delta_f, 1)