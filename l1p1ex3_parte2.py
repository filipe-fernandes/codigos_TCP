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

#####################################
#PRIMEIRA PARTE DO CÓGIGO:
#INICIALIZAÇÃO
#####################################
#PARAMETROS DA SIMULAÇÃO

t0=0
tf=10
dt=0.001
ts=0.1# do código anterior
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
g=np.loadtxt('Gvet.dat')
Ns=len(g)
N=Ns
M=Ns
pho=1

G0=np.zeros((N,N))
for i in range(0,N):
    for j in range(0,i+1):
        G0[j+N-i-1,j]=g[(N-i-1)]
G=G0[:,0:M]
print(G)

temp=np.zeros((1,M))
temp[0,0]=1
Kmpc=(temp@(np.linalg.inv(G.T@G+pho*np.eye(M,M))@G.T))[0,:]
# print(Kmpc)
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
p=0
ntem=0
acumulat=0
tempo_inicial=time.time()
for k in range(0,tamanho_t-1,1):
    t=t+dt
    ###
    #CONTROLE
    tc=time.perf_counter()
    if t>5:
        p=0.01
    if flag_ts>=razao_ts_dt:
        flag_ts=0
        for i in range(0,N-1):
            sum_n=0
            for n in range(0,Ns-1):
                if n+i<Ns:
                    g_temp=g[n+i]
                else:
                    g_temp=g[Ns-1]
                sum_n=sum_n+(g_temp-g[n])*DU[n]
            F[i,0]=y+sum_n
        du=(Kmpc@(Ref-F))[0]
        u=u+du
        DU=np.roll(DU,1)
        DU[0]=du
    acumulat+=(time.perf_counter()-tc)/10000
    # print(ntem,'demorou ','%.15f'%(time.perf_counter()-tc),'\n')
    ntem+=1
    ###
    #PLANTA 1/(S+1)
    y=(1-dt)*y+dt*u+p#perturbação dps de um tempo
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
plt.close('all')
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