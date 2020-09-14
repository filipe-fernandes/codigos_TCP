#####################################
#SIMULAÇÃO DO MOTOR DE INDUÇÃO CONTROLADO PELA REDE NEURAL
#O CÓDIGO É DIVIDIDO EM 3 PARTES: INICIALIZAÇÃO, SIMULAÇÃO E PLOTS
#####################################

#####################################
#IMPORTA OS PACOTES
#
#####################################

import numpy as np
import time
from math import sqrt
from math import pi
from math import sin
from math import cos
import matplotlib.pyplot as plt
np.random.seed(1)

print("começou\n")
tempo_inicial=time.time()
#####################################
#PRIMEIRA PARTE DO CÓGIGO:
#INICIALIZAÇÃO
#####################################
#PARAMETROS DA SIMULAÇÃO
dt = 1e-6
ts = 1e-3
razao_ts_dt=round(ts/dt)
tmax = 1-dt
npontos = round(tmax/dt)
i = 0

wb = 2*pi*60

# # #PARAMETROS DO MOTOR QUE A VANESSA TRABALHA
# rs = 20.6
# Lm = 0.78
# lls = 0.82425-Lm
# llr = 0.82425-Lm
# rr = 20.6
# jm = 0.012
# bm = 0.00881
# vrms = 220/sqrt(3)

rs = 0.435
xls = 0.754
xm = 26.13
xlr = 0.754
rr = 0.816
jm = 0.089
bm=0.00881
vrms = 220/sqrt(3)

lls = xls/wb
llr = xlr/wb
Lm = xm/wb

P = 4 #POLOS OU PAR DE POLOS?

Ls = Lm+lls
Lr = Lm+llr
sig = Lr*Ls-Lm**2

#VARIÁVEIS
vas = 0
vbs = 0
vcs = 0
valphas = 0
vbetas = 0

var = 0
vbr = 0
vcr = 0
valphar = 0
vbetar = 0

fas = 0
fbs = 0
fcs = 0
falphas = 0
fbetas = 0

far = 0
fbr = 0
fcr = 0
falphar = 0
fbetar = 0

ias = 0
ibs = 0
ics = 0
ialphas = 0
ibetas = 0
ids=0
iqs=0
izs=0

iar = 0
ibr = 0
icr = 0
ialphar = 0
ibetar = 0
idr=0
iqr=0

we = 0
wr = 0
wm = 0
te = 0
tl = 0
#tl = tb
oe = 0
om = 0
#####################

#ROTINAS DOS RUNGE KUTTA PARA OS FLUXOS E A VELOCIDADE
def rk_fs(vs,fs,fr):
    return vs-rs*(Lr*fs-Lm*fr)/sig

def rk_fr(vr,fr,fs,wr,frp):
    return vr-rr*(Ls*fr-Lm*fs)/sig+wr*frp

def dfluxos(valphas,vbetas,valphar,vbetar,falphas,fbetas,falphar,fbetar,rk_wr,rkts):
    k1=rkts*rk_fs(valphas,falphas,falphar)
    l1=rkts*rk_fs(vbetas,fbetas,fbetar)
    m1=rkts*rk_fr(valphar,falphar,falphas,-rk_wr,fbetar)
    n1=rkts*rk_fr(vbetar,fbetar,fbetas,rk_wr,falphar)

    k2=rkts*rk_fs(valphas,falphas+k1/2,falphar+m1/2)
    l2=rkts*rk_fs(vbetas,fbetas+l1/2,fbetar+n1/2)
    m2=rkts*rk_fr(valphar,falphar+m1/2,falphas+k1/2,-rk_wr,fbetar+n1/2)
    n2=rkts*rk_fr(vbetar,fbetar+n1/2,fbetas+l1/2,rk_wr,falphar+m1/2)

    k3=rkts*rk_fs(valphas,falphas+k2/2,falphar+m2/2)
    l3=rkts*rk_fs(vbetas,fbetas+l2/2,fbetar+n2/2)
    m3=rkts*rk_fr(valphar,falphar+m2/2,falphas+k2/2,-rk_wr,fbetar+n2/2)
    n3=rkts*rk_fr(vbetar,fbetar+n2/2,fbetas+l2/2,rk_wr,falphar+m2/2)

    k4=rkts*rk_fs(valphas,falphas+k3,falphar+m3)
    l4=rkts*rk_fs(vbetas,fbetas+l3,fbetar+n3)
    m4=rkts*rk_fr(valphar,falphar+m3,falphas+k3,-rk_wr,fbetar+n3)
    n4=rkts*rk_fr(vbetar,fbetar+n3,fbetas+l3,rk_wr,falphar+m3)
    return (k1+2*k2+2*k3+k4)/6,(l1+2*l2+2*l3+l4)/6,(m1+2*m2+2*m3+m4)/6,(n1+2*n2+2*n3+n4)/6

#VELOCIDADE ANGULAR
def frk_wm(wk,tek,tlk):
    return -bm/jm*wk+(tek-tlk)/jm

def rk_dwm(wk,trk,tek,tlk):
    k1=trk*frk_wm(wk,tek,tlk)
    k2=trk*frk_wm(wk+k1/2,tek,tlk)
    k3=trk*frk_wm(wk+k2/2,tek,tlk)
    k4=trk*frk_wm(wk+k3,tek,tlk)
    return 1/6*(k1+2*k2+2*k3+k4)

#CÁLCULO DAS TENSÕES DO INVERSOR (IGUAL AO CÓDIGO DA SABRINA)
def  inversor_ideal(s1, s2, s3, vdc):
    vas = (2.0/3.0*s1-1.0/3.0*s2-1.0/3.0*s3)*vdc
    vbs = (2.0/3.0*s2-1.0/3.0*s1-1.0/3.0*s3)*vdc
    vcs = (2.0/3.0*s3-1.0/3.0*s2-1.0/3.0*s1)*vdc
    return vas,vbs,vcs

#ROTINA PARA O CHAVEAMENTO (IGUAL AO CÓDIGO DA SABRINA)
def  chaveamento(indice_min):
    if indice_min ==1:
        sw1 = 1
        sw2 = 0
        sw3 = 0
    elif indice_min==2:
        sw1 = 0
        sw2 = 1
        sw3 = 0
    elif indice_min==3:
        sw1 = 1
        sw2 = 1
        sw3 = 0
    elif indice_min==4:
        sw1 = 0
        sw2 = 0
        sw3 = 1
    elif indice_min==5:
        sw1 = 1
        sw2 = 0
        sw3 = 1
    elif indice_min==6:
        sw1 = 0
        sw2 = 1
        sw3 = 1
    else:
        sw1 = 0
        sw2 = 0
        sw3 = 0
    return sw1, sw2, sw3

def abg_transform(a,b,c):
    alpha=(2*a-b-c)/3
    beta=sqrt(3)*(b-c)/3
    gamma=(a+b+c)/3
    return alpha,beta,gamma

def dq_transf(ias,ibs,ics,oe):
    ids=cos(oe)*ias+cos(oe-2*pi/3)*ibs+cos(oe+2*pi/3)*ics
    iqs=-sin(oe)*ias-sin(oe-2*pi/3)*ibs-sin(oe+2*pi/3)*ics
    izs=(ias+ibs+ics)/sqrt(3)
    return sqrt(2/3)*ids,sqrt(2/3)*iqs,izs

#INICIALIZAÇÃO DOS VETORES PARA O ARMEZENAMENTO DA INFORMAÇÃO
Vas = np.zeros(npontos)
Vbs = np.zeros(npontos)
Vcs = np.zeros(npontos)
Valphas = np.zeros(npontos)
Vbetas = np.zeros(npontos)

Var = np.zeros(npontos)
Vbr = np.zeros(npontos)
Vcr = np.zeros(npontos)
Valphar = np.zeros(npontos)
Vbetar = np.zeros(npontos)

Fas = np.zeros(npontos)
Fbs = np.zeros(npontos)
Fcs = np.zeros(npontos)
Falphas = np.zeros(npontos)
Fbetas = np.zeros(npontos)

Far = np.zeros(npontos)
Fbr = np.zeros(npontos)
Fcr = np.zeros(npontos)
Falphar = np.zeros(npontos)
Fbetar = np.zeros(npontos)

Ias = np.zeros(npontos)
Ibs = np.zeros(npontos)
Ics = np.zeros(npontos)
Ialphas = np.zeros(npontos)
Ibetas = np.zeros(npontos)
Ids = np.zeros(npontos)
Iqs = np.zeros(npontos)
Izs = np.zeros(npontos)

Iar = np.zeros(npontos)
Ibr = np.zeros(npontos)
Icr = np.zeros(npontos)
Ialphar = np.zeros(npontos)
Ibetar = np.zeros(npontos)

We = np.zeros(npontos)
Wr = np.zeros(npontos)
Wm = np.zeros(npontos)
Te = np.zeros(npontos)
Tl = np.zeros(npontos)
Oe = np.zeros(npontos)
Om = np.zeros(npontos)

T = np.zeros(npontos)
Sa= np.zeros(npontos)
Sb= np.zeros(npontos)
Sc= np.zeros(npontos)
Vrms = np.zeros(npontos)
Wb = np.zeros(npontos)
Wref = np.zeros(npontos)

#MAIS ALGUMAS VARIAVEIS ALTERADAS COM MAIS FREQUÊNCIA
wrefr=180#1500*pi/30
wref=0

for j in range(1,npontos):

    t=j*dt
    wref=wrefr*t/(0.3)
    if t>=3.9:
        wref=1.333*wrefr
    elif t>=3.6:
        wref=wrefr
    elif t>=3.3:
        wref=0.666*wrefr
    elif t>=3.0:
        wref=0.333*wrefr
    elif t>=2.8:
        wref=0
    elif t>=1.6:
        wref=2*wrefr
    elif t>=1.4:
        wref=0
    elif t>=1.0:
        wref=wrefr
    elif t>=0.9:
        wref=0
    elif t>=0.6:
        wref=wrefr-wrefr*(t-0.6)/(0.3)
    elif (t>=0.3):
        wref=wrefr
    
    Wref[j]=wref

j=0
tl = 0.1
sa=0
sb=0
sc=0

vdc=220*sqrt(2)*sqrt(3)
flag=5
Jmin=1e20
indice_min=0
flag2=0
n=1
print("inicialização concluída\n")
##############################################################################
#####################################
#SEGUNDA PARTE DO CÓGIGO:
#SIMULAÇÃO
#####################################
for t_int in range(npontos):
    t=t_int*dt
    ##########################################################
    ############################
    if t >= 0.7*tmax:
        tl = 0.5
    ##
    #ALIMENTAÇÃO DO SISTEMA
    if flag>=razao_ts_dt:#TESTE DO TEMPO DE CONTROLE
        #LOOPS DE CONTROLE UM PRA K+1E OUTRO PRA K+2
        #LOOPS DE CONTROLE UM PRA K+1E OUTRO PRA K+2
        Jmin=10000000000000000000000000000000000000000000
        for indice_1 in range(0,6):
            for indice_2 in range(0,6):
                s1,s2,s3=chaveamento(indice_1)
                vas_c,  vbs_c,  vcs_c =inversor_ideal(s1,s2,s3,vdc)
                #print("Tensões:\t",vas_c," ",vbs_c," ",vcs_c,"\n")#DEBUGS
                valphas_c = 2/3*(vas_c-1/2*vbs_c-1/2*vcs_c)
                vbetas_c = 2/3*(sqrt(3)/2*vbs_c-sqrt(3)/2*vcs_c)
                valphar_c = 0
                vbetar_c = 0

                dfalphas_c,dfbetas_c,dfalphar_c,dfbetar_c=dfluxos(valphas_c,vbetas_c,valphar_c,vbetar_c,falphas,fbetas,falphar,fbetar,wr,ts)
                dwm_c = rk_dwm(wm,ts,te,tl)

                #print("dfluxos",dfalphas_c," ",dfbetas_c," ",dfalphar_c," ",dfbetar_c," dwm",dwm_c," \n")#DEBUGS
                falphas_c = falphas+dfalphas_c
                fbetas_c = fbetas+dfbetas_c
                falphar_c = falphar+dfalphar_c
                fbetar_c = fbetar+dfbetar_c
                wm_c = wm+dwm_c
                wr_c = (P/2)*wm_c

                ialphas_c = (Lr*falphas_c-Lm*falphar_c)/sig
                ibetas_c = (Lr*fbetas_c-Lm*fbetar_c)/sig
                ialphar_c = (Ls*falphar_c-Lm*falphas_c)/sig
                ibetar_c = (Ls*fbetar_c-Lm*fbetas_c)/sig
                #print("correntes:",ialphas_c," ",ibetas_c," ",ialphar_c," ",ibetar_c," \n")#DEBUGS
                te_c = (3/2)*(P/2)*Lm*(ibetas_c*ialphar_c-ialphas_c*ibetar_c)
                #print("te_c:\t",te_c,"\n")#DEBUGS

                s1_2,s2_2,s3_2=chaveamento(indice_2)
                vas_c2,  vbs_c2,  vcs_c2 =inversor_ideal(s1_2,s2_2,s3_2,vdc)
                #print("Tensões:\t",vas_c," ",vbs_c," ",vcs_c,"\n")#DEBUGS
                valphas_c2 = 2/3*(vas_c2-1/2*vbs_c2-1/2*vcs_c2)
                vbetas_c2 = 2/3*(sqrt(3)/2*vbs_c2-sqrt(3)/2*vcs_c2)
                valphar_c2 = 0
                vbetar_c2 = 0

                dfalphas_c2,dfbetas_c2,dfalphar_c2,dfbetar_c2=dfluxos(valphas_c2,vbetas_c2,valphar_c2,vbetar_c2,falphas_c,fbetas_c,falphar_c,fbetar_c,wr_c,ts)
                dwm_c2 = rk_dwm(wm_c,ts,te_c,tl)

                #print("dfluxos",dfalphas_c," ",dfbetas_c," ",dfalphar_c," ",dfbetar_c," dwm",dwm_c," \n")#DEBUGS
                falphas_c2 = falphas_c+dfalphas_c2
                fbetas_c2 = fbetas_c+dfbetas_c2
                falphar_c2 = falphar_c+dfalphar_c2
                fbetar_c2 = fbetar_c+dfbetar_c2
                wm_c2 = wm_c+dwm_c2

                ialphas_c2 = (Lr*falphas_c2-Lm*falphar_c2)/sig
                ibetas_c2 = (Lr*fbetas_c2-Lm*fbetar_c2)/sig
                ialphar_c2 = (Ls*falphar_c2-Lm*falphas_c2)/sig
                ibetar_c2 = (Ls*fbetar_c2-Lm*fbetas_c2)/sig
                #print("correntes:",ialphas_c," ",ibetas_c," ",ialphar_c," ",ibetar_c," \n")#DEBUGS
                te_c2 = (3/2)*(P/2)*Lm*(ibetas_c2*ialphar_c2-ialphas_c2*ibetar_c2)
                #print("torque:",te_c2,"\n")#DEBUGS
                dwm_c3 = rk_dwm(wm_c2,ts,te_c2,tl)
                wm_c3=wm_c2+dwm_c3
                #print("wm:",wm_c3,"dwm:",dwm_c3,"\n")#DEBUGS
                pos_wref3=i+3*razao_ts_dt
                pos_wref2=i+2*razao_ts_dt
                if pos_wref3>=npontos :
                    pos_wref3=i
                    pos_wref2=i
                
                
                Jc=(Wref[pos_wref3]-wm_c3)**2+(Wref[pos_wref2]-wm_c2)**2
                #print("Jc:\t",Jc,"\n")#DEBUGS
                if Jc<=Jmin:
                    Jmin=Jc
                    indice_min=indice_1
        
        j+=1
        
        
        flag=0
        #EM CÓDIGOS QUE DEMORAM MUITO FICA DIFICIL SABER
        #SE TRAVOU OU SE TÁ RODANDO AINDA
        #ESSA LÓGICA MOSTRA QUANTOS % DO CÓDIGO PASSOU
        if flag2>=npontos*0.05*n/razao_ts_dt:
            print(n*5,"% concluído\n")
            n+=1
        flag2+=1
    #ENTRADA DA ALIMENTAÇÃO DO SISTEMA, QUE VEM DO CONTROLE
    sa,sb,sc=chaveamento(indice_min)
    vas,  vbs,  vcs =inversor_ideal(sa,sb,sc,vdc)
    we = wb#DESNECESSÁRIO
    #PASSAGEM PARA AS COORDENADAS ALPHA E BETA,NAQUAL O MOTOR É SIULADO
    valphas = 2/3*(vas-1/2*vbs-1/2*vcs)
    vbetas = 2/3*(sqrt(3)/2*vbs-sqrt(3)/2*vcs)
    valphar = 0
    vbetar = 0
    #CALCULO DOS DELTAS POR RUNGE KUTTA DE QUARTA ORDEM
    dfalphas,dfbetas,dfalphar,dfbetar=dfluxos(valphas,vbetas,valphar,vbetar,falphas,fbetas,falphar,fbetar,wr,dt)
    dwm = rk_dwm(wm,dt,te,tl)
    #SOMA DE K+dK PARA OBTER K+1
    falphas = falphas+dfalphas
    fbetas = fbetas+dfbetas
    falphar = falphar+dfalphar
    fbetar = fbetar+dfbetar
    wm = wm+dwm
    #CALCULO DAS CORRENTE
    ialphas = (Lr*falphas-Lm*falphar)/sig
    ibetas = (Lr*fbetas-Lm*fbetar)/sig
    ialphar = (Ls*falphar-Lm*falphas)/sig
    ibetar = (Ls*fbetar-Lm*fbetas)/sig
    #TORQUE ELÉTRICO
    te = (3/2)*(P/2)*Lm*(ibetas*ialphar-ialphas*ibetar)
    #O TORQUE SÓ ALTERARÁ A VELOCIDADE NO PRÓXIMO LOOP
    #FIM DA SIMULAÇÃO
    ##############################################
    #AS EXPRESSÕES A SEGUIR APENAS TRATAM AS VARIAVEIS PARA POSTERIOR LEITURA
    ias = ialphas
    ibs = -1/2*ialphas+sqrt(3)/2*ibetas
    ics = -1/2*ialphas-sqrt(3)/2*ibetas

    iar = ialphar*sin(0-oe)+ibetar*cos(0-oe)
    ibr = ialphar*sin(0-oe-2*pi/3)+ibetar*cos(0-oe-2*pi/3)
    icr = ialphar*sin(0-oe-4*pi/3)+ibetar*cos(0-oe-4*pi/3)

    fas = falphas
    fbs = -1/2*falphas+sqrt(3)/2*fbetas
    fcs = -1/2*falphas-sqrt(3)/2*fbetas

    far = falphar*sin(-oe)+fbetar*cos(-oe)
    fbr = falphar*sin(-oe-2*pi/3)+fbetar*cos(-oe-2*pi/3)
    fcr = falphar*sin(-oe-4*pi/3)+fbetar*cos(-oe-4*pi/3)

    wr = (P/2)*wm #WR É USADO PARA O CÁLCULO DO FLUXO NO ROTOR
    oe = oe+dt*wr
    if oe > (P/2)*2*pi:
        oe = oe-(P/2)*2*pi

    if oe < 0:
        oe = oe+(P/2)*2*pi
    ids,iqs,izs=dq_transf(ias,ibs,ics,oe)
    idr,iqr,izr=dq_transf(iar,ibr,icr,oe)
    ##########################################################################
    #LEITURA DAS VARIÁVEIS
    Sa[i]=sa
    Sb[i]=sb
    Sc[i]=sc

    Vas[i] = vas
    Vbs[i] = vbs
    Vcs[i] = vcs
    Valphas[i] = valphas
    Vbetas[i] = vbetas

    Var[i] = var
    Vbr[i] = vbr
    Vcr[i] = vcr
    Valphar[i] = valphar
    Vbetar[i] = vbetar

    Fas[i] = vas
    Fbs[i] = vbs
    Fcs[i] = vcs
    Falphas[i] = valphas
    Fbetas[i] = vbetas

    Far[i] = far
    Fbr[i] = fbr
    Fcr[i] = fcr
    Falphar[i] = falphar
    Fbetar[i] = fbetar

    Ias[i] = ias
    Ibs[i] = ibs
    Ics[i] = ics
    Ialphas[i] = ialphas
    Ibetas[i] = ibetas
    Ids[i] = ids
    Iqs[i] = iqs
    Izs[i] = izs

    Iar[i] = iar
    Ibr[i] = ibr
    Icr[i] = icr
    Ialphar[i] = ialphar
    Ibetar[i] = ibetar

    We[i] = we
    Wr[i] = wr
    Wm[i] = wm
    Te[i] = te
    Tl[i] = tl
    Oe[i] = oe

    T[i] = t

    i = i+1
    flag+=1


print('demorou ',time.time()-tempo_inicial,' s\n\n\n')
#####################################
#TERCEIRA PARTE DO CÓGIGO:
#PLOT E/OU SALVAR OS DADOS
#####################################

plt.rc('text', usetex=False)
#plt.rc('font', family='serif')

plt.figure()
plt.plot(T,Sa,color='C0',label=r'Sa')
plt.plot(T,Sb,color='C1',label=r'Sb')
plt.plot(T,Sc,color='C2',label=r'Sc')
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$Estado das chaves$")
plt.title(r"Ação de Controle")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(T,Ialphas,color='C0',label=r'Ias')
plt.plot(T,Ibetas,color='C1',label=r'Ibs')
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$i[A]$")
plt.title(r"Correntes estator")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(T,Ialphar,color='C0',label=r'Iar')
plt.plot(T,Ibetar,color='C1',label=r'Ibr')
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$i[A]$")
plt.title(r"Correntes rotor")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(Wm,Te,color='C0')
plt.xlabel(r"$Speed$[rpm]")
plt.ylabel(r"Torque")
plt.title(r"Angular Speed")
plt.grid()
plt.legend()
plt.show()


plt.figure()
plt.plot(T,Wref,color='k',label=r'Reference')
plt.plot(T,Wm,color='C0',label=r'ts=5e-3')
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$\omega (t)$ [rad/s]")
plt.title(r"Angular Speed")
plt.grid()
plt.legend()
plt.show()


#CODE CEMETERY

# figure()
# plot(T,Wm,label=R'wm')
# xlabel(R"$t$[s]")
# ylabel(R"$v(t)$ [V]")
# title(r"Tensao")
# grid()
# legend()
# show()

#
# display(plot(T,[Iar,Ibr,Icr], legend = false))
# display(plot(T,[Vas,Vbs,Vcs], legend = false))
# display(plot(Wm*30/pi,Te, legend = false))
# plot(T,Wref, legend = false,color=:black)
# plot!(T,Wm,color=RGB(0, 0.85, 0.85))


# dfalphas_c = rk_dfalphas(valphas_c,falphas,falphar,ts)
# dfbetas_c = rk_dfbetas(vbetas_c,fbetas,fbetar,ts)
# dfalphar_c = rk_dfalphar(valphar_c,falphar,falphas,fbetar,ts)
# dfbetar_c = rk_dfbetar(vbetar_c,fbetar,fbetas,falphar,ts)

# dfalphas = rk_dfalphas(valphas,falphas,falphar,dt)
# dfbetas = rk_dfbetas(vbetas,fbetas,fbetar,dt)
# dfalphar = rk_dfalphar(valphar,falphar,falphas,fbetar,dt)
# dfbetar = rk_dfbetar(vbetar,fbetar,fbetas,falphar,dt)


#ROTINA PARA O CHAVEAMENTO (IGUAL AO CÓDIGO DA SABRINA)
# def  chaveamento(indice_min):
#     # local sw1, sw2, sw3
#     if indice_min ==1:
#         sw1 = 1
#         sw2 = 0
#         sw3 = 0
#     elif indice_min==2
#         sw1 = 0
#         sw2 = 1
#         sw3 = 0
#     elif indice_min==3
#         sw1 = 1
#         sw2 = 1
#         sw3 = 0
#     elif indice_min==4
#         sw1 = 0
#         sw2 = 0
#         sw3 = 1
#     elif indice_min==5
#         sw1 = 1
#         sw2 = 0
#         sw3 = 1
#     elif indice_min==6
#         sw1 = 0
#         sw2 = 1
#         sw3 = 1
#     else
#         sw1 = 0
#         sw2 = 0
#         sw3 = 0
    
#     return sw1, sw2, sw3

