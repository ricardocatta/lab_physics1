from cProfile import label
from re import T
import statistical_module as sm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Importa os dados coletados do experimento usando o porta pesos fixoe variando o taco
ppfixo = pd.read_csv("pp_fixo.csv", keep_default_na=True) #importa a lista de dados

# Importa os dados coletados do experimento para o relatório 1
tacofixo = pd.read_csv("taco_fixo.csv", keep_default_na=True) #importa a lista de dados



# Criando um vetor com as massas do portapeso
mp = np.array([tacofixo.m_p1, tacofixo.m_p1, tacofixo.m_p1, tacofixo.m_p1, 
tacofixo.m_p1, tacofixo.m_p1, tacofixo.m_p1]) / 1000

ei_mp = np.ones(7) / 1000

# Criando um vetor com as massas do portapeso
mpp = np.array([ppfixo.m_p1, ppfixo.m_p2, ppfixo.m_p3, ppfixo.m_p4, 
ppfixo.m_p5, ppfixo.m_p6, ppfixo.m_p7]) / 1000
ei_mpp = np.ones(7) / 1000

# Criando um vetor com as massas do taco
mt = np.array([tacofixo.m_t1, tacofixo.m_t2, tacofixo.m_t3, tacofixo.m_t4, 
tacofixo.m_t5, tacofixo.m_t6, tacofixo.m_t7]) / 1000

ei_mt = np.ones(7) / 1000

# Criando um vetor com as massas do taco
mtp = np.array([ppfixo.m_t1, ppfixo.m_t1, ppfixo.m_t1, ppfixo.m_t1, 
ppfixo.m_t1, ppfixo.m_t1, ppfixo.m_t1]) / 1000

ei_mtp = np.ones(7) / 1000


# Criando um vetor com a massa total fixando a massa do taco
m_tot = mp + mt

# Criando um vetor com a massa total fixando a massa do portapeso
m_totp = mpp + mtp

ei_m_totp = np.ones(7) / 1000


at = np.array([tacofixo.a1, tacofixo.a2, tacofixo.a3, tacofixo.a4, 
tacofixo.a5, tacofixo.a6, tacofixo.a7])



ei_at = np.array([[1, 1, 0.01, 0.01, 0.1, 0.01, 1],
[1, 1, 0.01, 0.01, 0.01, 1, 0.01], 
[0.01, 0.01,  1, 0.01, 1, 0.01, 0.01], 
[0.01, 0.01, 0.01, 1, 0.01, 0.01, 0.01], 
[0.1, 0.01, 1, 0.01, 1, 0.01, 0.01], 
[0.01, 1, 0.01, 0.01, 0.01, 1, 0.01], 
[1, 0.01, 0.01, 0.01, 0.01, 0.01, 1]])



app = np.array([ppfixo.a1, ppfixo.a2, ppfixo.a3, ppfixo.a4, 
ppfixo.a5, ppfixo.a6, ppfixo.a7])

ei_app = np.array([[1, 1, 0.01, 0.01, 0.1, 0.01, 1],
[1, 1, 0.01, 0.01, 0.01, 1, 0.01], 
[0.01, 0.01,  1, 0.01, 1, 0.01, 0.01], 
[0.01, 0.01, 0.01, 1, 0.01, 0.01, 0.01], 
[0.1, 0.01, 1, 0.01, 1, 0.01, 0.01], 
[0.01, 1, 0.01, 0.01, 0.01, 1, 0.01], 
[1, 0.01, 0.01, 0.01, 0.01, 0.01, 1]])




mt_mean = np.zeros(7)
for i in range(7):
    mt_mean[i] = sm.mean(m_tot[i], 4)



ei_mt_mean = np.zeros(7)
for i in range(7):
    ei_mt_mean[i] = sm.mean(ei_at[i], 4)



mp_mean = np.zeros(7)
for i in range(7):
    mp_mean[i] = sm.mean(m_totp[i], 4)



ei_mp_mean = np.zeros(7)
for i in range(7):
    ei_mp_mean[i] = sm.mean(ei_mp[i], 4)


at_mean = np.zeros(7)
for i in range(7):
    at_mean[i] = sm.mean(at[i], 4)



ap_mean = np.zeros(7)
for i in range(7):
    ap_mean[i] = sm.mean(app[i], 4)



ea_mt = np.zeros(7)
for i in range(7):
    ea_mt[i] = sm.erro_associado(m_tot[i], 4, ei_mt[i])



ei_at_mean = np.zeros(7)
for i in range(7):
    ei_at_mean[i] = sm.mean(ei_at[i], 4)


ea_mp = np.zeros(7)
for i in range(7):
    ea_mp[i] = sm.erro_associado(m_totp[i], 4, ei_mp[i])



ei_app_mean = np.zeros(7)
for i in range(7):
    ei_app_mean[i] = sm.mean(ei_app[i], 4)


ea_at = np.zeros(7)
for i in range(7):
    ea_at[i] = sm.erro_associado(at[i], 4, ei_at_mean[i])




ea_app = np.zeros(7)
for i in range(7):
    ea_app[i] = sm.erro_associado(app[i], 4, ei_app_mean[i])

print("at_mean", np.mean(at_mean))
print("mp_mean", np.mean(mp))
print("mt_mean", np.mean(mt))



ln_at = np.log(at_mean)
ln_mt = np.log(mt_mean)

m1, b1 = sm.least_square(ln_mt, ln_at, 4)

propaga_inc_at = sm.propaga_incerteza_3D(1/at_mean, 0,0,ea_at, 0, 0)

propaga_inc_mt = sm.propaga_incerteza_3D(1/mt_mean, 0,0,ea_mt, 0, 0)


ln_app = np.log(ap_mean)
ln_mpp = np.log(mp_mean)

m2, b2 = sm.least_square(ln_mpp, ln_app, 4)

propaga_inc_app = sm.propaga_incerteza_3D(1/ap_mean, 0,0,ea_app, 0, 0)

propaga_inc_mpp = sm.propaga_incerteza_3D(1/mp_mean, 0,0,ea_mp, 0, 0)


def plot_lnm_lna_tacofixo(m, b, x, y, delx, dely):
    """
    Plota o gráfico para a altura h1.

    OUTPUT:

    Retorna o gráfico com os erros associados e a reta linearizada pelo MMQ.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('$\ln{a}$ $[m/s²]$')
    axes1.set_xlabel('$\ln{m}$ $[kg]$')
    x1 = x
    y1 = y
    y2 = b + m * x1
    y2 = np.array(y)
    print("\n coeficiente linear do gráfico lnm vs lna = \n", b)
    print("\n coeficiente angular do gráfico lnm vs lna = \n", m)
    print("\n incerteza do coeficiente angular = \n", sm.sigma_m(propaga_inc_at, ln_mt))
    print("\n incerteza do coeficiente linear = \n", sm.sigma_b(propaga_inc_at, ln_mt))
    print("\n k = \n", np.exp(b2))
    print("\n ln_a = \n", y)
    print("\n ln_m = \n", x)
    plt.plot(x1, y2, '-', label ="linearização por MMQ - taco fixo.\n")
    print("\n erro associado da aceleração com o taco fixo (ea_at) = \n", ea_at)
    print("\n erro instrumental da massa média com o taco fixo (ei_mt_mean) = \n", ei_mt_mean)
    print("\n erro associado da massa com o taco fixo (ea_mt) = \n", ea_mt)
    print("\n aceleração média com o taco fixo (at_mean) = \n", at_mean)
    print("\n erro instrumental da massa média com o taco fixo (ei_mt_mean) = \n", ei_mt_mean)
    print("\n massa total média com o taco fixo (mt_mean) = \n", mt_mean)
    print("\n erro instrumental com o taco fixo (ei_at) = \n", ei_at)
    print("\n aceleração com o taco fixo (at) = \n", at)
    print("\n massa total com o taco fixo (m_tot) = \n", m_tot)
    print("\n massa do taco com o taco fixo (mt) = \n", mt)
    print('\n massa do pp com o taco fixo (mp) = \n', mp)
    
    
    #plt.plot(x1, y1, '*', label ="ln dos pontos")
    #Coloca a barra de erro%
    ls = ''
    for i in range(len(y)):
        if i == 0:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P1")  
        elif i == 1:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P2")
        elif i == 2:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P3")
        elif i == 3:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P4")
        elif i == 4:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P5")
        elif i == 5:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P6")
        else:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P7")  
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()


def plot_lnm_lna_ppfixo(m, b, x, y, delx, dely):
    """
    Plota o gráfico para a altura h1.

    OUTPUT:

    Retorna o gráfico com os erros associados e a reta linearizada pelo MMQ.

    """
    plt.style.use('ggplot')
    fig = plt.figure(dpi=110)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('$\ln{a}$ $[m/s²]$')
    axes1.set_xlabel('$\ln{m}$ $[kg]$')
    x1 = x
    y1 = y
    y = b + m * x
    y = np.array(y)
    print("\n coeficiente linear do gráfico lnm vs lna = \n", b)
    print("\n coeficiente angular do gráfico lnm vs lna = \n", m)
    print("\n incerteza do coeficiente angular ln_mpp = \n", sm.sigma_m(propaga_inc_app, ln_mpp))
    print("\n incerteza do coeficiente linear ln_mpp = \n", sm.sigma_b(propaga_inc_app, ln_mpp))
    print("\n k = \n", np.exp(b2))
    print("\n ln_a = \n", y1)
    print("\n ln_m = \n", x1)
    plt.plot(x, y, '-', label ="linearização por MMQ - pp fixo. \n")
    print("\n erro associado da aceleração com o pp fixo (ea_app) = \n", ea_app)
    print("\n erro instrumental da massa média com o pp fixo (ei_app_mean) = \n", ei_app_mean)
    print("\n erro associado da massa com o pp fixo (ea_mp) = \n", ea_mp)
    print("\n aceleração média com o pp fixo (ap_mean) = \n", ap_mean)
    print("\n erro instrumental da massa média com o pp fixo (ei_mp_mean) = \n", ei_mp_mean)
    print("\n massa total média com o pp fixo (mp_mean) = \n", mp_mean)
    print("\n aceleração com o pp fixo (app) =\n", app)
    print("\n erro instrumental com o pp fixo (ei_app) = \n", ei_app)
    print("\n massa total com o pp fixo (m_totp) = \n", m_totp)
    print("\n massa do pp com o pp fixo (mpp) = \n", mpp)
    print("\n massa do taco com o pp fixo (mtp) = \n", mtp)
    
    
    #plt.plot(x1, y1, '*', label ="ln dos pontos")
    #Coloca a barra de erro%
    ls = ''
    for i in range(len(y)):
        if i == 0:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P1")  
        elif i == 1:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P2")
        elif i == 2:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P3")
        elif i == 3:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P4")
        elif i == 4:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P5")
        elif i == 5:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P6")
        else:
            plt.errorbar(x1[i], y1[i], xerr=delx[i],  yerr=dely[i], linestyle=ls, marker='o',label ="ponto experimental P7")  
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()



plot_lnm_lna_ppfixo(m2, b2, ln_mpp, ln_app, propaga_inc_mpp, propaga_inc_app)
#plot_lnm_lna_tacofixo(m1, b1, ln_mt, ln_at, propaga_inc_mt, propaga_inc_at)
print("\n erro relativo = \n", 100 * abs(4.99 - 5)/5)
