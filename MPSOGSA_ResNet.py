## MPSOGSA
import train_base_model as ResNetBasics
import os
import torch
import numpy as np
from numpy import random as rnd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import sys
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt
import argparse  
#import SaveDataCsv as SV

def euclid_dist(x,y):
    temp = 0   
    for i,j in zip(x,y):
        temp += (i-j)**2
        final = np.sqrt(temp)
    return final
def PSOGSA_ResNet(dataset,max_iters,num_particles,NumSave,lr,savepath):
    np.seterr(divide='ignore', invalid='ignore')
    # %config InlineBackend.figure_format = 'retina'
    c1 = 2
    c2 = 2
    g0 = 1
    dim =2
    w1=2;                 
    wMax=0.9            
    wMin=0.5              
    current_fitness = np.zeros((num_particles,1))
    gbest = np.zeros((1,dim))
    gbest_score = float('inf')
    OldBest=float('inf')

    convergence = np.zeros(max_iters)
    alpha = 20
    epsilon = 1

    class Particle:
        pass

    #all particle initialized
    particles = []
    for i in range(num_particles):
        p = Particle()
        alpha=np.random.uniform(0.5,0.9)
        p.params=[np.random.randint(300*0.4,300),alpha]
        Width=[]
        tmpOld=np.random.randint(3072*alpha,3072)
        Numlayers=3
        for k in range(Numlayers):
            tmpNew=np.random.randint(tmpOld*alpha,tmpOld)
            tmpOld=tmpNew
            Width.append(tmpNew)
       

        p.fitness = rnd.rand()
        p.velocity = 0.3*rnd.randn(dim)
        p.res_force = rnd.rand()
        p.acceleration = rnd.randn(dim)
        p.force = np.zeros(dim)
        p.id = i
        particles.append(p)

    #training 
    print('training begain:', dataset)
    for i in range(max_iters):
        if i % 10 == 0:
            print('iteration number:', i)
        # gravitational constant
        g = g0*np.exp((-alpha*i)/max_iters)
        # calculate mse
        cf = 0    
        for p in particles:
            fitness = 0
            y_train = 0
            if p.params[0]<100 or p.params[0]>300:
                p.params[0]=np.random.randint(300*0.4,300)
            elif p.params[1]<0.4 or p.params[1]>0.9:
                p.params[2]=np.random.uniform(0.4,0.9)
            Width=[]
            tmpOld=np.random.randint(3072*alpha,3072)
            for k in range(Numlayers):
                tmpNew=np.random.randint(tmpOld*alpha,tmpOld)
                tmpOld=tmpNew
                Width.append(tmpNew)    
            print('hidden size, number of layers, and scale are:',p.params)
            fitness = ResNetBasics.ResNet(dataset,Width,lr,savepath)
            hiddensize=int(p.params[0])
            numlayers=int(p.params[1])
            
    #         fitness = fitness/X.shape[0]
            OldFitness=fitness
            current_fitness[cf] = fitness
            cf += 1
            if gbest_score > fitness and OldBest>fitness:
                """hiddenState=np.array(hidden0.view(numlayers,hiddensize).tolist())
                rp = RecurrencePlot()
                X_rp = rp.fit_transform(hiddenState)
                plt.figure(figsize=(6, 6))
                plt.imshow(X_rp[0], cmap='binary', origin='lower')
            #         plt.title('Recurrence Plot', fontsize=14)
                plt.savefig(savepath+'/RecurrencePlots/'+'RecurrencePlots_'+dataset+str(round(fitness,NumSave))+'_'
                                +str(numlayers)+'_'+str(hiddensize)+'_'
                                       +'.png',dpi=600)
                plt.show()

                
                weightsName='reservoir.weight_hh'
                for name, param in named_parameters:
        #             print(name,param)
                    if name.startswith(weightsName):
        #                 set_trace()
                        torch.save(param,savepath+'weights'+str(round(fitness,6))+'.pt') """
                OldBest=gbest_score
                gbest_score = fitness
                gbest = p.params

        best_fit = min(current_fitness)
        worst_fit = max(current_fitness)

        for p in particles:
            p.mass = (current_fitness[particles.index(p)]-0.99*worst_fit)/(best_fit-worst_fit)

        for p in particles:
            p.mass = p.mass*5/sum([p.mass for p in particles])


        # gravitational force
        for p in particles:
            for x in particles[particles.index(p)+1:]:
                p.force = (g*(x.mass*p.mass)*(p.params - x.params))/(euclid_dist(p.params,x.params))

        # resultant force
        for p in particles:
            p.res_force = p.res_force+rnd.rand()*p.force

        # acceleration
        for p in particles:
            p.acc = p.res_force/p.mass

        w1 = wMin-(i*(wMax-wMin)/max_iters)

        # velocity
        for p in particles:
            p.velocity = w1*p.velocity+rnd.rand()*p.acceleration+rnd.rand()*(gbest - p.params)

        # position
        for p in particles:
            p.params = p.params + p.velocity

        convergence[i] = gbest_score
#     set_trace()  
    plt.figure(figsize=(6, 6))
    plt.plot(convergence)  
    plt.xlabel('Convergence')
    plt.ylabel('Error')
    plt.draw()
    plt.savefig(savepath+dataset+args.PredictionRatio+'_ConvergenceChanges.png',dpi=600)  

    sys.stdout.write('\rMPSOGSA is training ESN (Iteration = ' + str(i+1) + ', MSE = ' + str(gbest_score) + ')')
    sys.stdout.flush()
        # save results 
    FileName=dataset+'_BestParameters.csv'
    newdata=[args.PredictionRatio,max_iters,num_particles,p.params,convergence]
    PathFileName=os.path.join(savepath,FileName)
    SV.SaveDataCsv(PathFileName,newdata)


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

parser.add_argument('--dataset',default='CIFAR10',type=str, help='dataset to train')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('--max_iters', type=int,default=50, help='')

parser.add_argument('--num_particles', type=int, default=30, help='')

parser.add_argument('--NumSave', type=int, default=8, help='Number to save')

parser.add_argument('--savepath', type=str,required=False, default='../Results/',
                    help='Path to save results')

args = parser.parse_args()

if __name__ =="__main__":
    torch.cuda.is_available()
    PSOGSA_ResNet(args.dataset,args.max_iters,args.num_particles,args.NumSave,args.lr,args.savepath)

