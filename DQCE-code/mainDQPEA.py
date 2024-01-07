# coding:utf-8
import copy

from DataRead import DataReadDHFJSP
import numpy as np
import os
from Initial import GHInitial
from inital import initial
from fitFJSP import CalfitDHFJFP
from Tselection import *
from EA import evolution
from Tool import *
from FastNDSort import FastNDS
from EnergySave import EnergysavingDHFJSP
from LocalSearch import *
from DQN_model import DQN
import torch

Combination=[[10,2],[20,2],[30,2],[40,2],\
             [20,3],[30,3],[40,3],[50,3],\
             [40,4],[50,4],[100,4],\
             [50,5],[100,5],[150,5],\
             [100,6],[150,6],[200,6],\
             [100,7],[150,7],[200,7]]

datapath='../DATASET/'
FileName=[];ResultPath=[]
for i in range(20):
    J=Combination[i][0]
    F1=Combination[i][1]
    O=5
    temp = datapath +  str(J) +'J' + str(F1)+ 'F' + '.txt'
    temp2 = str(J) +'J' + str(F1)+ 'F'
    FileName.append(temp);
    ResultPath.append(temp2)
TF=20
FileName=np.array(FileName);FileName.reshape(TF,1)
ResultPath=np.array(ResultPath);ResultPath.reshape(TF,1)
#read the parameter of algorithm such as popsize, crossover rate, mutation rate
f= open("parameter.txt", "r", encoding='utf-8')
ps,Pc,Pm,lr,batch_size,EPSILON,GAMMA,MEMORY_CAPACITY = f.read().split(' ')
ps=int(ps);Pc=float(Pc);Pm=float(Pm);lr=float(lr);batch_size=int(batch_size)
EPSILON=float(EPSILON);GAMMA=float(GAMMA);MEMORY_CAPACITY=int(MEMORY_CAPACITY)
IndependentRun=10

lr=0.001;batch_size=16;
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
MEMORY_CAPACITY = 512

TARGET_REPLACE_ITER = 7   # target update frequency
N_ACTIONS = 9  # 4种�?选的算子
EPOCH=1


print(torch.cuda.is_available())

#execute algorithm for each instance
CCF=19
for file in range(CCF,CCF+1):
    N,F,TM,H,SH,NM,M,time,ProF=DataReadDHFJSP(FileName[file])
    MaxNFEs=200*SH
    #create filepath to store the pareto solutions set for each independent run
    respath='DQNV9+ES\\';sprit='\\'
    respath=respath+ResultPath[file]
    isExist=os.path.exists(respath)
    #if the result path has not been created
    if not isExist:
        currentpath=os.getcwd()
        os.makedirs(currentpath+sprit+respath)
    print(ResultPath[file],'is being Optimizing\n')
    #start independent run for GMA
    for rround in range(1):
        p_chrom,m_chrom,f_chrom=initial(N,H,SH,NM,M,ps,F)
        fitness=np.zeros(shape=(ps,3))
        NFEs=0 #number of function evaluation
        #calucate fitness of each solution
        for i in range(ps):
            fitness[i,0],fitness[i,1],fitness[i,2]=CalfitDHFJFP(p_chrom[i,:],m_chrom[i,:],f_chrom[i,:],N,H,SH,F,TM,time)

        AP=[];AM=[];AF=[];AFit=[]# Elite archive
        i=1
        #build model
        N_STATES=2*SH+N
        CountOpers = np.zeros(N_ACTIONS)
        PopCountOpers = []
        dq_net = DQN(N_STATES, N_ACTIONS, BATCH_SIZE=batch_size, LR=lr, EPSILON=EPSILON, GAMMA=GAMMA, \
                     MEMORY_CAPACITY=MEMORY_CAPACITY, TARGET_REPLACE_ITER=TARGET_REPLACE_ITER)
        Loss=[]
        while NFEs<MaxNFEs:
            print(FileName[file]+' round ',rround+1,'iter ',i)
            i = i + 1
            ChildP = np.zeros(shape=(2 * ps, SH), dtype=int)
            ChildM = np.zeros(shape=(2 * ps, SH), dtype=int)
            ChildF = np.zeros(shape=(2 * ps, N), dtype=int)
            ChildFit = np.zeros(shape=(2 * ps, 3))
            # mating selection
            P_pool, M_pool, F_pool = tournamentSelection(p_chrom, m_chrom,f_chrom, fitness, ps, SH, N)
            # offspring generation
            for j in range(ps):
                Fit1=np.zeros(3);Fit2=np.zeros(3);
                P1, M1,F1, P2, M2,F2 = evolution(P_pool, M_pool,F_pool, j, Pc, Pm, ps, SH, N, H, NM, M)
                Fit1[0],Fit1[1],Fit1[2] = CalfitDHFJFP(P1, M1,F1, N,H,SH,F,TM,time)
                Fit2[0],Fit2[1],Fit2[2] = CalfitDHFJFP(P2, M2,F2, N,H,SH,F,TM,time)
                NFEs = NFEs + 2;
                t1 = j * 2;
                t2 = j * 2 + 1
                ChildP[t1, :] = copy.copy(P1);ChildM[t1, :] = copy.copy(M1);ChildF[t1, :] = copy.copy(F1);ChildFit[t1, :] = Fit1
                ChildP[t2, :] = copy.copy(P2);ChildM[t2, :] = copy.copy(M2);ChildF[t2, :] = copy.copy(F2);ChildFit[t2, :] = Fit2
            QP = np.vstack((p_chrom, ChildP))
            QM = np.vstack((m_chrom, ChildM))
            QF = np.vstack((f_chrom, ChildF))
            QFit = np.vstack((fitness, ChildFit))
            QP, QM, QF,QFit = DeleteReapt(QP, QM, QF,QFit, ps)
            RQFit=QFit[:,0:2]
            TopRank = FastNDS(RQFit, ps)
            p_chrom = QP[TopRank, :];
            m_chrom = QM[TopRank, :];
            f_chrom = QF[TopRank, :];
            fitness = QFit[TopRank, :]

            PF = pareto(fitness)
            if len(AFit) == 0:
                AP = copy.copy(p_chrom[PF, :])
                AM = copy.copy(m_chrom[PF, :])
                AF = copy.copy(f_chrom[PF, :])
                AFit = copy.copy(fitness[PF, :])


            # Elite strategy
            PF = pareto(fitness)
            if len(AFit) == 0:
                AP = p_chrom[PF, :]
                AM = m_chrom[PF, :]
                AF = f_chrom[PF, :]
                AFit = fitness[PF, :]
            else:
                AP = np.vstack((AP, p_chrom[PF, :]))
                AM = np.vstack((AM, m_chrom[PF, :]))
                AF = np.vstack((AF, f_chrom[PF, :]))
                AFit = np.vstack((AFit, fitness[PF, :]))
            PF = pareto(AFit)
            AP = AP[PF, :];
            AM = AM[PF, :];
            AF = AF[PF, :];
            AFit = AFit[PF, :];
            AP,AM,AF,AFit= DeleteReaptE(AP, AM, AF,AFit)

            #Local search in Archive

            L=len(AFit)
            current_state = np.zeros(N_STATES,dtype=int)
            next_state = np.zeros(N_STATES, dtype=int)
            for l in range(L):
                current_state[0:SH]=copy.copy(AP[l,:])
                current_state[SH:SH*2]=copy.copy(AM[l,:])
                current_state[SH*2:N_STATES]=copy.copy(AF[l,:])

                action = dq_net.choose_action(current_state)
                k=int(action)
                if k == 0:
                    P1, M1, F1 = N6(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F)
                elif k == 1:
                    P1, M1, F1 = SwapOF(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time)
                elif k == 2:
                    P1, M1, F1 = RandFA(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F)
                elif k == 3:
                    P1, M1, F1 = RandMS(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F)
                elif k == 4:
                    P1, M1, F1 = InsertOF(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time)
                elif k == 5:
                    P1, M1, F1 = InsertIF(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time,F)
                elif k == 6:
                    P1, M1, F1 = SwapIF(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time,F)
                elif k == 7:
                    P1, M1, F1 = RankFA(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F,ProF)
                elif k == 8:
                    P1, M1, F1 = RankMS(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F)

                Fit1[0], Fit1[1], Fit1[2] = CalfitDHFJFP(P1, M1, F1, N, H, SH, F, TM, time)
                NFEs = NFEs + 1
                dom=NDS(Fit1, AFit[l, :])
                if  dom== 1:
                    AP[l, :] = copy.copy(P1);
                    AM[l, :] = copy.copy(M1);
                    AF[l, :] = copy.copy(F1);
                    AFit[l, :] = copy.copy(Fit1)
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))
                    reward=5
                elif dom == 0 and AFit[l][0]!=Fit1[0] and AFit[l][1]!=Fit1[1]:
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))
                    reward = 10
                else:
                    reward=0
                next_state[0:SH] = copy.copy(P1)
                next_state[SH:SH * 2] = copy.copy(M1)
                next_state[SH * 2:N_STATES] = copy.copy(F1)
                dq_net.store_transition(current_state, action, reward, next_state)
                if dq_net.memory_counter > 50:
                    for epoch in range(EPOCH):
                        loss=dq_net.learn()
                        Loss.append(loss)

            # Energy save
            L = len(AFit)
            for j in range(L):
                P1,M1,F1=EnergysavingDHFJSP(AP[j,:],AM[j,:],AF[j,:],AFit[j,:],N,H,TM,time,SH,F)
                Fit1[0], Fit1[1], Fit1[2] = CalfitDHFJFP(P1, M1, F1, N, H, SH, F, TM, time)
                NFEs=NFEs+1
                if NDS(Fit1,AFit[j,:])==1:
                    AP[j, :]=copy.copy(P1);
                    AM[j, :]=copy.copy(M1);
                    AF[j, :]=copy.copy(F1);
                    AFit[j,:]=copy.copy(Fit1)
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))
                elif NDS(Fit1,AFit[j,:])==0:
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))


        #write elite solutions in txt
        PF = pareto(AFit)
        AP = AP[PF, :];
        AM = AM[PF, :];
        AF = AF[PF, :];
        AFit = AFit[PF, :];
        PF=pareto(AFit)
        l=len(PF)
        obj=AFit[:,0:2]
        newobj=[]
        for i in range(l):
            newobj.append(obj[PF[i],:])
        newobj=np.unique(newobj,axis=0)# delete the repeat row
        tmp='res'
        resPATH=respath+sprit+tmp+str(rround+1)+'.txt'
        f=open(resPATH, "w", encoding='utf-8')
        l=len(newobj)
        for i in range(l):
            item='%5.2f %6.2f\n'%(newobj[i][0],newobj[i][1])# fomat writing into txt file
            f.write(item)
        f.close()

        f = open('loss.txt', "w", encoding='utf-8')
        l = len(Loss)
        for i in range(l):
            item = '%f\n' % (Loss[i])  # fomat writing into txt file
            f.write(item)
        f.close()

    print('finish '+FileName[file])
print('finish running')
