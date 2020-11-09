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
from control import sample_system
from control import TransferFunction

print("começou\n")
tempo_inicial=time.time()
#####################################
#PRIMEIRA PARTE DO CÓGIGO:
#INICIALIZAÇÃO
#####################################
#PARAMETROS DA SIMULAÇÃO
sysc = TransferFunction([1], [1,1])
sysd = sample_system(sysc, 0.1, method='zoh')

t0=0
tf=10
dt=0.1
tamanho_t=round((tf-t0)/dt)

#INICIALIZAÇÃO DOS VETORES PARA O ARMEZENAMENTO DA INFORMAÇÃO
T=np.zeros(tamanho_t)
U=np.zeros(tamanho_t)
Y=np.zeros(tamanho_t)
R=np.zeros(tamanho_t)

#VARIÁVEIS
u=1
t=0
y=0
yt=0
r=1

#SETANDO CONDIÇÕES INICIAIS
T[0]=0
Y[0]=0
U[0]=1
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
    y=0.09516 *U[k] +0.9048* Y[k]
    if y>=0.63*u and yt==0:
        yt=y
        tau=t
    ##########################################################################
    #LEITURA DAS VARIÁVEIS
    Y[k+1]=y
    U[k+1]=u
    T[k+1]=t
    R[k+1]=r

#tau tem que ser menor que o negócio
#olhar a amostragem direito
N=3
M=3
# pho=1.9
Ntau=1
G0=np.zeros((N,N))
for i in range(0,N):
    for j in range(0,i+1):
        G0[j+N-i-1,j]=Y[Ntau*(N-i)]
G=G0[:,0:M]
print(G)
# np.savetxt('G2.dat',G)
i=1
g_temp=np.zeros(100)

while True:
    g_temp[i]=Y[Ntau*i]
    i+=1
    if abs(Y[-1] - g_temp[i-1])/Y[-1]<0.02:#2%
        break
print(g_temp)
g=g_temp[np.concatenate((np.where(g_temp!=0)),axis=None)]
# np.savetxt('Gvet2.dat',g)
Ns=len(g)
temp=np.zeros((1,M))
temp[0,0]=1
# Kmpc=(temp@(np.linalg.inv(G.T@G+pho*np.eye(M,M))@G.T))[0,:]
# print(Kmpc)



print('demorou ',time.time()-tempo_inicial,' s\n\n\n')
#####################################
#TERCEIRA PARTE DO CÓGIGO:
#PLOT E/OU SALVAR OS DADOS
#####################################
plt.rc('text', usetex=False)

plt.figure()
plt.plot(T, U,color='C0', label=r"$Entrada$")
plt.plot(T, Y,color='C1', label=r"$Saída$")
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$V(t)[V]$")
plt.title(r"Formas de onda dos Sinais")
plt.grid()
plt.legend()
plt.show()

