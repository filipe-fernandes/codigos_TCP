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
tf=100
dt=0.0001
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
    
    u=1
    
    ###
    #PLANTA 1/(S+1)
    y=(1-dt)*y+dt*u
    if y>=0.63 and yt==0:
        yt=y
        tau=t
    ##########################################################################
    #LEITURA DAS VARIÁVEIS
    Y[k+1]=y
    U[k+1]=u
    T[k+1]=t
    R[k+1]=r

#Vai virar um DMC dps
N=3
M=3
pho=1.9
Ntau=round(tau/dt)
G0=np.zeros((N,N))
for i in range(0,N):
    for j in range(0,i+1):
        G0[j+N-i-1,j]=Y[Ntau*(N-i)]
G=G0[:,0:M]
print(G)
np.savetxt('G.dat',G)
i=1
Gvet=np.zeros(100)

while True:
    Gvet[i]=Y[Ntau*i]
    i+=1
    if Gvet[i-1] -Gvet[i-2]<0.00000001 :
        break
print(Gvet)
Gvetout=Gvet[np.concatenate(([0],np.where(Gvet!=0)),axis=None)]
np.savetxt('Gvet.dat',Gvetout)
temp=np.zeros((1,M))
temp[0,0]=1
Kmpc=(temp@(np.linalg.inv(G.T@G+pho*np.eye(M,M))@G.T))[0,:]
print(Kmpc)



print('demorou ',time.time()-tempo_inicial,' s\n\n\n')
#####################################
#TERCEIRA PARTE DO CÓGIGO:
#PLOT E/OU SALVAR OS DADOS
#####################################
plt.rc('text', usetex=False)
#plt.rc('font', family='serif')
'''
plt.figure()
plt.plot(T, U,color='C0', label=r"$Entrada$")
plt.plot(T, Y,color='C1', label=r"$Saída$")
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$V(t)[V]$")
plt.title(r"Formas de onda dos Sinais")
plt.grid()
plt.legend()
plt.show()
'''
