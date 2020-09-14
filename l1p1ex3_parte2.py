# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:22:46 2020

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
import math
import time

print("começou\n")
tempo_inicial=time.time()
#####################################
#PRIMEIRA PARTE DO CÓGIGO:
#INICIALIZAÇÃO
#####################################
#PARAMETROS DA SIMULAÇÃO

t0=0
tf=20
dt=0.01
ts=(0.9942999999999068)/5
razao_ts_dt=round(ts/dt)
tamanho_t=round((tf-t0)/dt)

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
du=0

#Matrizes de controle
N=3
M=3
pho=10.01

G=np.loadtxt('G.dat')
Gvet=np.loadtxt('Gvet.dat')
print(G)
temp=np.zeros((1,M))
temp[0,0]=1
Kmpc=(temp@(np.linalg.inv(G.T@G+pho*np.eye(M,M))@G.T))[0,:]
print(Kmpc)

Ns=len(Gvet)-N
DU=np.zeros(Ns)
#SETANDO CONDIÇÕES INICIAIS
T[0]=0
Y[0]=0
U[0]=0
R[0]=0

Ref=np.ones((N,1))*r
F=np.zeros((N,1))
flag_ts=0
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
    if flag_ts>=razao_ts_dt:
        flag_ts=0
        for i in range(0,N):
            sum_n=0
            for n in range(0,Ns):
                sum_n=sum_n+(Gvet[n+i]-Gvet[n])*DU[n]
            F[i,0]=y+sum_n
        du=(Kmpc@(Ref-F))[0]
        u=u+du
        DU=np.roll(DU,1)
        DU[0]=du
    ###
    #PLANTA 1/(S+1)
    y=(1-dt)*y+dt*u
    ##########################################################################
    #LEITURA DAS VARIÁVEIS
    Y[k+1]=y
    U[k+1]=u
    T[k+1]=t
    R[k+1]=r
    flag_ts+=1


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
plt.title(r"Ação de controle")
plt.grid()
plt.legend()
plt.show()