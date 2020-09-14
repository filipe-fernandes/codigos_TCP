# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:04:31 2020

@author: cliente
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 08:49:33 2020

@author: cliente
"""
#####################################
#SIMULAÇÃO DO SISTEMA 1/(S+1)
#O CÓDIGO É DIVIDIDO EM 3 PARTES: INICIALIZAÇÃO, SIMULAÇÃO E PLOTS
#####################################

#####################################
#IMPORTA OS PACOTES
#
#####################################

import numpy as np
import matplotlib.pyplot as plt
import time

print("começou\n")
tempo_inicial=time.time()
#####################################
#PRIMEIRA PARTE DO CÓGIGO:
#INICIALIZAÇÃO
#####################################
#PARAMETROS DA SIMULAÇÃO

t0=0
tf=10
dt=0.0001
tamanho_t=round((tf-t0)/dt)
kp=0.1
ki=1e-3

#INICIALIZAÇÃO DOS VETORES PARA O ARMEZENAMENTO DA INFORMAÇÃO
T=np.zeros(tamanho_t)
U=np.zeros(tamanho_t)
Y=np.zeros(tamanho_t)
R=np.zeros(tamanho_t)

#VARIÁVEIS
u=10
t=0
y=0
yt=0
r=1
ea=0
#SETANDO CONDIÇÕES INICIAIS
T[0]=0
Y[0]=0
U[0]=0
R[0]=0

print("inicialização concluída\n")
##############################################################################
#####################################
#SEGUNDA PARTE DO CÓGIGO:
#SIMULAÇÃO
#####################################
for k in range(0,tamanho_t-1,1):
    t=t+dt
    ###
    #CONTROLE
    e=r-y
    ea=e+ea
    u=kp*e+ki*ea
    
    ###
    #PLANTA 1/(S+1)
    y=(1-dt)*y+dt*u
    
    ##########################################################################
    #LEITURA DAS VARIÁVEIS
    Y[k+1]=y
    U[k+1]=u
    T[k+1]=t
    R[k+1]=r

print('demorou ',time.time()-tempo_inicial,' s\n\n\n')
#####################################
#TERCEIRA PARTE DO CÓGIGO:
#PLOT E/OU SALVAR OS DADOS
#####################################
plt.rc('text', usetex=False)
#plt.rc('font', family='serif')
plt.figure()
plt.plot(T, R,color='k', label=r"$Referência$")
plt.plot(T, Y,color='C1', label=r"$Saída$")
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$V(t)[V]$")
plt.title(r"Formas de onda dos Sinais")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(T, U,color='C0', label=r"$Entrada$")
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$V(t)[V]$")
plt.title(r"Formas de onda dos Sinais")
plt.grid()
plt.legend()
plt.show()

