#CONVERGENCE THROUGH INFERENCE VER 6.0
#Author: Phanish Puranam (phanish.puranam@insead.edu)
#with probability q of joint reference after failure.

#Importing modules
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from scipy.stats import entropy


STARTINGTIME = datetime.datetime.now().replace(microsecond=0)

#####################################################################################################
# SET SIMULATION PARAMETERS HERE
T=1000#number of periods to simulate the model
NP=100#number of pairs of agents
alternate1=1#if 1, then agents alternate as Sender and Receiver
alternate2=1#if 1, then agents alternate as Trainer and Trainee
q=0#probability of revealing stimulus after failure
print ("alternate1="+str(alternate1)+"alternate2="+str(alternate2)+", q="+str(q))

#Communication code stuctures
M1=10#rows in code
M2=20#columns size of code matrix. M2>M1: implies initial synonymy (row sharpness>col sharpness initially), M2<M1  homonymy (row sharpness<col sharpness)
signalvariety=M2-1#how many unique signals the environment generates, [0,M2-1]
a1=0.0 #initial ambiguity [0,1]
a2=0.0 #initial ambiguity [0,1]
amb1 = int((a1+0.1)*10) #  initial ambiguity in the code of the first individual, from 1-11 (translates into a=0.1(ambi-1))
amb2 = int((a2+0.1)*10) #  initial ambiguity in the code of the second individual, from 1-11
lowval=0.0#to replace lowest value instead of 0 with a non zero number


#agent learning parameters
tau=0.01#softmax temperature
phi1=0.5#learning rate in Reinforcement learning for agent 1; same value used for both kinds of update
phi2=0.5#learning rate in Reinforcement learning for agent 2; same value used for both kinds of update
phi12=1*phi1#Agent 1 updates based on other's choice of label/stimulus or stimulus/label
phi21=1*phi2#Agent 2 updates based on other's choice of label/stimulus or stimulus/label
######################################################################################################

#agent payoffs
P1s=1#payoff from correctly identifying stimulus, agent 1 as sender
p1s=0#payoff from failing to identify stimulus, agent 1 as sender
P2s=1#payoff from correctly identifying stimulus,agent 2 as sender
p2s=0#payoff from gailing to identify stimulus,agent 2 as sender

P1r=1#payoff from correctly identifying stimulus, agent 1 as receiver
p1r=0#payoff from failing to identify stimulus, agent 1 as receiver
P2r=1#payoff from correctly identifying stimulus,agent 2 as receiver
p2r=0#payoff from failing to identify stimulus,agent 2 as receiver

#DEFINING RESULTS VECTORS
#finally we will produce per agent, one vector of length T and breadth=2 that reports average across NP pairs of agents per period and cumulative performance  
avg_agent1_perf=np.zeros((T,NP))
avg_agent2_perf=np.zeros((T,NP))
switch1=np.zeros((T,NP))
switch2=np.zeros((T,NP))
avg_org_perf=np.zeros((T,NP))
avg_org_cumperf=np.zeros((T,NP))
divergence1=np.zeros((T,NP))
divergence2=np.zeros((T,NP))
divergence3=np.zeros((T,NP))
divergence4=np.zeros((T,NP))
avg_agent1_cumperf=np.zeros((T,NP))
avg_agent2_cumperf=np.zeros((T,NP))
avg_org_str_beliefR1 = np.zeros((T, NP))  
avg_org_str_beliefR2 = np.zeros((T, NP))
avg_org_str_beliefC1 = np.zeros((T, NP)) 
avg_org_str_beliefC2 = np.zeros((T, NP))
state = np.zeros((T, NP))

#Defining functions
def language0(m1,m2):#builds an initial code with all zeros
    r=np.zeros((m1,m2))
    return r

def languageFlat(m1,m2):#builds an initial code with all zeros
    r=(1/2)*np.ones((m1,m2))
    return r

def language(ambi):# builds eaach agents's version of the code
    
    r = 0.1*(np.random.choice(int(ambi),(M1,M2),replace=True))

    if M1<M2:    
        order=np.random.choice(int(M2),(1,M2),replace=False)
        for j in range(M1):
            a1=order[0,j]
            r[j, a1]=1
    
    if M2<M1:    
        order=np.random.choice(int(M1),(1,M1),replace=False)
        for j in range(M2):
            a1=order[0,j]
            r[a1, j]=1
            
    if M1==M2:    
        order=np.random.choice(int(M2),(1,M2),replace=False)
        for j in range(M2):
            a1=order[0,j]
            r[a1, j]=1
            
    return r
    
def languageNewUL(ambi,agent):# builds each agent's initial version of the code with unseen labels
    r1=int(M1/2)
    r2=int(M2/2)
    r = 0.1*(np.random.choice(int(ambi),(M1,M2),replace=True))
    #r= np.zeros((M1,M2))+lowval#toactivate lowval

    if agent==1:
        order=np.random.choice(r1,(1,r1),replace=False)
        for j in range(r1):
            a1=order[0,j]
            r[j, a1]=1
            
        for j in range(r1,M1):
            for k in range(M2):
                r[j,k]=lowval#new
    
    if agent==2:
        order=np.random.choice(r1,(1,r1),replace=False)
        for j in range(r1,M1):
            a1=order[0,j-r1]
            r[j, a1]=1
            
        for j in range(r1):
            for k in range(M2):
                r[j,k]=lowval#new
    
    return r
   
    
def languageNewUS(ambi,agent):#builds each agent's initial version of the code with unseen stimuli
    r1=int(M1/2)
    r2=int(M2/2)
    r = 0.1*(np.random.choice(int(ambi),(M1,M2),replace=True))
    #r= np.zeros((M1,M2))+lowval#to activate lowval
    
    if agent==1:
        order=np.random.choice(r2,(1,r2),replace=False)
        for j in range(r2):
            a1=order[0,j]
            r[a1, j]=1
            
        for j in range(r2,M2):
            for k in range(M1):
                r[k,j]=lowval#new
    
    if agent==2:
        order=np.random.choice(r2,(1,r2),replace=False)
        for j in range(r2,M2):
            a1=order[0,j-r2]
            r[a1,j]=1
            
        for j in range(r2):
            for k in range(M1):
                r[k,j]=lowval#new
    
    return r


def languageNewULUS(ambi,agent):# Crisper ULUS (Jan 2019) builds each agent's version of the language with unseen labels and stimuli
    r1=int(M1/2)
    r2=int(M2/2)
    r = 0.1*(np.random.choice(int(ambi),(M1,M2),replace=True))
    #r= np.zeros((M1,M2))+lowval#to activate lowval
    
    if agent==1:
        order=np.random.choice(r2,(1,r2),replace=False)
        for j in range(r2):
            a1=order[0,j]
            r[a1, j]=1
        
                 
        for j in range(r1):
            for k in range(r2,M2):
                r[j,k]=lowval#new
                
        for j in range(r1,M1):
            for k in range(0,M2):
                r[j,k]=lowval#new
    
    if agent==2:  
        order=np.random.choice(int(r2),(1,r2),replace=False)
        for j in range(r2,M2):
            a1=order[0,j-r2]+r2
            r[j, a1]=1
            
        for j in range(r1):
            for k in range(M2):
                r[j,k]=lowval#new
                
        for j in range(r1,M1):
            for k in range(r2):
                r[j,k]=lowval#new
    
    return r

def languagesparelabelsULUS(ambi,agent):# adds extra labels at bottom of code
    r1=int(2*M1/3)
    r2=int(M2/2)
    r = 0.1*(np.random.choice(int(ambi),(M1,M2),replace=True))
    #r= np.zeros((M1,M2))+lowval#to activate lowval
    
    if agent==1:
        order=np.random.choice(r2,(1,r2),replace=False)
        for j in range(r2):
            a1=order[0,j]
            r[a1, j]=1
        
                 
        for j in range(r1):
            for k in range(r2,M2):
                r[j,k]=lowval#new
                
        for j in range(r1,M1):
            for k in range(0,M2):
                r[j,k]=lowval#new
    
    if agent==2:  
        order=np.random.choice(int(r2),(1,r2),replace=False)
        for j in range(r2,M2):
            a1=order[0,j-r2]+r2
            r[j, a1]=1
            
        #for j in range(r1):
         #   for k in range(M2):
          #      r[j,k]=lowval#new
                
        for j in range(r1,M1):
            for k in range(M2):
                r[j,k]=lowval#new
    
    return r  
    
    
    
    r= np.zeros((M1+spares,M2))#to add extra labels
    for i in range (M1):
        for j in range (M2):
            r[i,j]=language[i,j]
  
    return r



def rsimilarity(a1,a2):# measures convergence on rows across codes
    dbeliefs=np.zeros(M1)    
    for row in np.arange(M1):
        max1=a1[row,:].argmax(axis=0)
        max2=a2[row,:].argmax(axis=0)
        if max1==max2:
            dbeliefs[row]=1
    return float(np.sum(dbeliefs))/M1


def csimilarity(a11,a22):#measures convergence on columns across codes
    
    a1=np.transpose(a11)
    a2=np.transpose(a22)
    dbeliefs=np.zeros(M2)    
    for row in np.arange(M2):
        max1=a1[row,:].argmax(axis=0)
        max2=a2[row,:].argmax(axis=0)
        if max1==max2:
            dbeliefs[row]=1
    return float(np.sum(dbeliefs))/M2

def rcossimilarity(a1,a2):# measures convergence on rows across codes using cosine similarity
    dbeliefs=np.zeros(M1)    
    for row in np.arange(M1):
        max1=a1[row,:]
        max2=a2[row,:]
        dot=np.dot(max1,max2)
        norm1 = np.linalg.norm(max1)
        norm2=np.linalg.norm(max2)
        cos=dot/(norm1*norm2)
        dbeliefs[row]=cos
    return float(np.sum(dbeliefs))/M1


def ccossimilarity(a11,a22):# measures convergence on columns across codes using cosine similarity
    a1=np.transpose(a11)
    a2=np.transpose(a22)
    dbeliefs=np.zeros(M2)    
    for row in np.arange(M2):
        max1=a1[row,:]
        max2=a2[row,:]
        dot=np.dot(max1,max2)
        norm1 = np.linalg.norm(max1)
        norm2=np.linalg.norm(max2)
        cos=dot/(norm1*norm2)
        dbeliefs[row]=cos
    return float(np.sum(dbeliefs))/M2


def str_beliefR(mat): #measures crispness within rows
    beliefs = np.zeros(M1)
    for row1 in np.arange(M1):
        labels_beliefs = np.sort(mat[row1])  # sorted from lowest to highest
        beliefs[row1] = labels_beliefs[M2-1] - labels_beliefs[M2-2]
    return np.mean(beliefs)

def str_beliefC(mat):#measures crispness within columns
    beliefs = np.zeros(M2)
    for col1 in np.arange(M2):
        labels_beliefs = np.sort(mat[:,col1])  # sorted from lowest to highest
        beliefs[col1] = labels_beliefs[M1-1] - labels_beliefs[M1-2]
    return np.mean(beliefs)

def estr_beliefR(mat): #measures crispness within rows using entropy
    beliefs = np.zeros(M1)
    for row1 in np.arange(M1):
        norm=np.linalg.norm(mat[row1])        
        beliefs[row1] = entropy(mat[row1]/norm, base=2)
    return np.mean(beliefs)

def estr_beliefC(mat):#measures crispness within columns using entropy
    beliefs = np.zeros(M2)
    for col1 in np.arange(M2):
        norm=np.linalg.norm(mat[col1])
        beliefs[col1] = entropy(mat[col1]/norm, base=2)
        
    return np.mean(beliefs)



def dissimilarity(l1,l2):
    diff=l1-l2    
    c=float(np.sum(np.absolute(diff),axis=None)/((np.shape(l1)[0])*(np.shape(l1)[0])))
    return c

def hardmax(attraction,t,dim): #max action selection
    maxcheck=attraction[t,:].max(axis=0)
    mincheck=attraction[t,:].min(axis=0)
    if maxcheck==mincheck:
        choice=random.randint(0, dim-1)    
    else:
        choice=attraction[t,:].argmax(axis=0)
    return choice

def softmax(attraction,t,dim): #softmax action selection

    prob=np.zeros((1,dim))
    denom=0
    i=0
    while i<dim:
        denom=denom + math.exp((attraction[t][i])/tau)
        i=i+1
    roulette=random.random()
    i=0
    p=0
    while i<dim:
        prob[0][i]=math.exp(attraction[t][i]/tau)/denom
        p= p+ prob[0][i]
        if p>roulette:
            choice= i
            return choice
            break #stops computing probability of action selection as soon as cumulative probability exceeds roulette        
        i=i+1
    
def checklabel(labelpick,stim,language):#if picked label already has an association,returns 1
    labels_beliefs = np.sort(language[labelpick])
    if labels_beliefs[M2-1]>language[labelpick,stim]:
        check1=1
    else:
        check1=0
    return check1
        
def checkstimulus(stimpick,lab,language):#if picked label already has an association,returns 1
    stim_beliefs = np.sort(language[:,stimpick])
    if stim_beliefs[M1-1]>language[lab,stimpick]:
        check2=1
    else:
        check2=0
    return check2   

def checksoftmaxlabel1(dim):

    i=0
    while i<dim:
        lint=softmax(att1C,t,M1)
        checking=checklabel(lint,s1,l1)
        if checking==0:
            return lint
            break
        i=i+1
    return lint
           
def checksoftmaxstim2(dim):

    i=0
    while i<dim:
        sint=softmax(att2R,t,M2)
        checking=checkstimulus(sint,l,l2)
        if checking==0:
            return sint
            break
        i=i+1
    return sint
    
def checksoftmaxlabel2(dim):

    i=0
    while i<dim:
        lint=softmax(att2C,t,M1)
        checking=checklabel(lint,s2,l2)
        if checking==0:
            return lint
            break
        i=i+1
    return lint

def checksoftmaxstim1(dim):

    i=0
    while i<dim:
        sint=softmax(att1R,t,M2)
        checking=checkstimulus(sint,l,l1)
        if checking==0:
            return sint
            break
        i=i+1
    return sint

#Time varying model objects
att1C=np.zeros((T,M1))
att2C=np.zeros((T,M1))
att1R=np.zeros((T,M2))
att2R=np.zeros((T,M2))

#To keep track of c(c)ount of # times action sleected for bayesian updating
cl1=np.ones((M1,M2))
cl2=np.ones((M1,M2))
pay1=np.zeros((M1,M2))
pay2=np.zeros((M1,M2))


#SIMULTAION IS RUN HERE

for a in range(NP):

    #l1 = language(amb1)#Baseline
    #l2 = language(amb2)#Baseline
    #l1=language0(M1,M2)#Creates codes with all zeros
    #l2=language0(M1,M2)#Creates codes with all zeros
    #l1=languageFlat(M1,M2)#Creates codes with all 1/2
    #l2=languageFlat(M1,M2)#Creates codes with all 1/2
    #l1=languageNewUL(amb1,1)#USE ONLY WHEN M1/2<=M2
    #l2=languageNewUL(amb2,2)#USE ONLY WHEN M1/2<=M2
    l1=languageNewUS(amb1,1)#USE ONLY WHEN M2/2<=M1
    l2=languageNewUS(amb2,2)#USE ONLY WHEN M2/2<=M1
    #l1=languageNewULUS(amb1,1)#USE ONLY WHEN M1=M2=EVEN NUMBER
    #l2=languageNewULUS(amb2,2)#USE ONLY WHEN M1=M2=EVEN NUMBER
    #l1=languagesparelabelsULUS(amb1,1)    
    #l2=languagesparelabelsULUS(amb2,2)   
    

    l10=np.copy(l1)#storing immutable copy of initial language
    l20=np.copy(l2)
    #pay1=np.copy(l1)
    #pay2=np.copy(l2)
    pay1=l10
    pay2=l20
    cl1=np.ones((M1,M2))
    cl2=np.ones((M1,M2))
    

 #   print np.round(l1,1)
#    print ("Initial")
#    print (l1)
#    print ()
#    print (l2)
#    print ()
    
  #  print np.round(l2,1)


    for t in range(T):
        
        if (alternate1*(t%2))==0:#to allow alternation in Sender/Receiver roles            
            s1=random.randint(0, signalvariety)
            att1C[t] = l1[:, s1]
            l=softmax(att1C,t,M1)#softmax 
            #l=hardmax(att1C,t,M1)#hardmax
            #l=checksoftmaxlabel1(M1)#homonymy avoidance
            #cl1[l,s1]=cl1[l,s1] + 1#for Bayesian

            sender=1
                       
            att2R[t] = l2[l, :]
            s2=softmax(att2R,t,M2)#softmax
            #s2=hardmax(att2R,t,M2)#hardmax
            #s2=checksoftmaxstim2(M2)#synonymy avoidance
            #cl2[l,s2]=cl2[l,s2]+1#for Bayesian
           
            
                      
        else:
            s2=random.randint(0, signalvariety)
            att2C[t] = l2[:, s2]
            l=softmax(att2C,t,M1)#softmax
            #l=hardmax(att2C,t,M1)#hardmax
            #l=checksoftmaxlabel2(M1)# homonymy avoidance
            #cl2[l,s2]=cl2[l,s2]+1#for Bayesian
           
            sender=2
         
            att1R[t] = l1[l, :]
            s1=softmax(att1R,t,M2)#softmax
            #s1=hardmax(att1R,t,M2)#hardmax
            #s1=checksoftmaxstim1(M2)#synoymy avoidance
            #cl1[l,s1]=cl1[l,s1]+1#for Bayesian
           
            
        if s1==s2:
            if sender==1:
                payoff1=P1s
                payoff2=P2r
            else:
                payoff1=P1r
                payoff2=P2s
        else:
            if sender==1:
                payoff1=p1s
                payoff2=p2r
            else:
                payoff1=p1r
                payoff2=p2s
        
        if (alternate2*(t%2))==0:#to allow alternation in Trainer/Trainee roles
            phi11=phi1
            phi22=phi2
            phi12=phi1
            phi21=phi2
        else:
            phi11=phi2
            phi22=phi1
            phi12=phi2
            phi21=phi1
    
          
        avg_agent1_perf[t][a]=payoff1      
        avg_agent2_perf[t][a]=payoff2
        #pay1[l][s1]=pay1[l][s1]+payoff1#for bayesian
        #pay2[l][s2]=pay2[l][s2]+payoff2#for bayesian

  
                
        
        if payoff1==0:
            roulette=random.random()
            if roulette<q:
               l1[l][s2]=l1[l][s2]+ phi12*(1-l1[l][s2])#label swapping
               l2[l][s1]=l2[l][s1]+ phi21*(1-l2[l][s1])#label swapping
               l1[l][s1]=l1[l][s1]+ phi11*(payoff1-l1[l][s1])#label swapping
               l2[l][s2]=l2[l][s2]+ phi22*(payoff2-l2[l][s2])#label swapping  
            else:
                l1[l][s1]=l1[l][s1]+ phi11*(payoff1-l1[l][s1])#RL updating
                l2[l][s2]=l2[l][s2]+ phi22*(payoff2-l2[l][s2])#RL updating
                #l1[l][s1]=pay1[l][s1]/cl1[l][s1]#bayesian
                #l2[l][s2]=pay2[l][s2]/cl2[l][s2]#bayesian

                
#                
        else:
            l1[l][s1]=l1[l][s1]+ phi11*(payoff1-l1[l][s1])#RL updating
            l2[l][s2]=l2[l][s2]+ phi22*(payoff2-l2[l][s2])#RL updating
            #l1[l][s1]=pay1[l][s1]/cl1[l][s1]#bayesian
            #l2[l][s2]=pay2[l][s2]/cl2[l][s2]#bayesian


        divergence1[t][a]=rsimilarity(l1,l2) 
        divergence2[t][a]=csimilarity(l1, l2)
        divergence3[t][a]=dissimilarity(l1,l10)
        divergence4[t][a]=dissimilarity(l2, l20)
        avg_org_str_beliefR1[t, a] = str_beliefR(l1)
        avg_org_str_beliefR2[t, a] = str_beliefR(l2)
        avg_org_str_beliefC1[t, a] = str_beliefC(l1)
        avg_org_str_beliefC2[t, a] = str_beliefC(l2)
        state[t,a]=np.abs(avg_agent1_perf[t,a]-avg_agent1_perf[t-1,a])#measure of instability of system
      

#        print ("at end of period" + str(t))
#        print ("pay")
#        print (pay1)
#        print
#        print (pay2)
#        print
#        print("L")
#        print (l1)
#        print
#        print (l2)
#        print
        
    
##PRODUCE RESULTS
#print ("Final")
##print (np.round(l1,2))
#print ("l1",l1)
#print ()
#print ("l2",l2)
##print()
##print (np.round(l2,2))
#print ()
#print("pay1",pay1)
#print ("pay2",pay2)

result_org=np.zeros((T,12))

for t in range(T):#Compiling final output
    result_org[t,0]=t
    result_org[t,1]=float(np.sum(avg_agent1_perf[t,:])+np.sum(avg_agent2_perf[t,:]))/(2.0*NP)
    result_org[t,2]=2*np.sum(result_org[:,1])    
    result_org[t,3]=float(np.sum(avg_org_str_beliefR1[t,:]))/NP
    result_org[t,4]=float(np.sum(avg_org_str_beliefC1[t,:]))/NP
    result_org[t,5]=float(np.sum(avg_org_str_beliefR2[t,:]))/NP
    result_org[t,6]=float(np.sum(avg_org_str_beliefC2[t,:]))/NP
    result_org[t,7]=float(np.std((avg_agent1_perf[t,:]+avg_agent2_perf[t,:]/2.0)))
    result_org[t,8]=float(np.sum(divergence1[t,:]))/NP
    result_org[t,9]=float(np.sum(divergence2[t,:]))/NP
    result_org[t,10]=float(np.sum(divergence3[t,:]))/NP-float(np.sum(divergence4[t,:]))/NP
    result_org[t,11]=float(np.sum(state[t,:]))/NP

#WRITING RESULTS TO CSV FILE   
#filename = ("GeneralizedInference5.7Rect"+"RL_Lc_Phi01" + "_M1_"+str(M1)+"_M2_"+str(M2) + "_phi_" + str((phi1+phi2)/2) + "_delta_" + str((phi1-phi2)/2) + "_T_" + str(T) + "_NP_" + str(NP)+'.csv')
#with open (filename,'w',newline='')as f:
#    thewriter=csv.writer(f)
#    thewriter.writerow(['Period', 'Performance', 'Cumlative Performance','BeliefStrengthR1','BeliefStrengthC1','BeliefStrengthR2','BeliefStrengthC2','StdDev of Match','Row Similarity','Col Similarity','DiffAdapt','Instability'])
#    for values in result_org:
#        thewriter.writerow(values)
#    f.close()  


##PRINTING END RUN RESULTS
print ("Final Match "+str(result_org[T-1,1]))
print ("Final Similarity "+str((result_org[T-1,8]+result_org[T-1,9])/2.0))
print ("Final Crispness "+str((result_org[T-1,3]+result_org[T-1,5]+result_org[T-1,4]+result_org[T-1,6])/4.0))



#
##GRAPHICAL OUTPUT
plt.style.use('ggplot')  # Setting the plotting style
fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
ax1 = plt.subplot2grid((6, 5), (4, 1), colspan=3)

ax1.plot(result_org[:, 0], (result_org[:, 3]+result_org[:,5])/2.0 , color='black',linestyle='--', label='Avg. S|L Crispness')
ax1.plot(result_org[:, 0], (result_org[:, 4]+result_org[:,6])/2.0, color='blue',linestyle='--', label='Avg L|S Crispness')


ax1.plot(result_org[:, 0], (result_org[:, 3])/1.0 , color='black',linestyle='--', label='A1. S|L Crispness')
ax1.plot(result_org[:, 0], (result_org[:, 4])/1.0, color='blue',linestyle='--', label='A1 L|S Crispness')
ax1.plot(result_org[:, 0], (result_org[:, 5])/1.0 , color='red',linestyle='--', label='A2. S|L Crispness')
ax1.plot(result_org[:, 0], (result_org[:, 6])/1.0, color='yellow',linestyle='--', label='A2 L|S Crispness')


ax1.set_xlabel('t', fontsize=12)
ax1.legend(bbox_to_anchor=(0., -0.8, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=9)

fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
ax2 = plt.subplot2grid((6, 5), (4, 1), colspan=3)

ax2.plot(result_org[:, 0], result_org[:, 8] , color='black', label='Avg S|L Similarity')
ax2.plot(result_org[:, 0], result_org[:, 9] , color='green', label='Avg L|S Similarity')


#x2.set_xlabel('t', fontsize=12)
#x2.legend(bbox_to_anchor=(0., -0.8, 1., .102), loc=3,
#          ncol=2, mode="expand", borderaxespad=0., fontsize=9)
           
fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
                 
ax3 = plt.subplot2grid((6, 5), (4, 1), colspan=3)
ax3.plot(result_org[:, 0], result_org[:, 1] , color='black', label='Matches')
ax3.set_xlabel('t', fontsize=12)
ax3.legend(bbox_to_anchor=(0., -0.8, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=9)
           

fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
                 
ax3 = plt.subplot2grid((6, 5), (4, 1), colspan=3)
ax3.plot(result_org[:, 0], result_org[:, 10] , color='black', label='Difference in Code Adaptation (Agent1-Agent2)')
ax3.set_xlabel('t', fontsize=12)
ax3.legend(bbox_to_anchor=(0., -0.8, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=9)

fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
                 
ax3 = plt.subplot2grid((6, 5), (4, 1), colspan=3)
ax3.plot(result_org[:, 0], result_org[:, 11] , color='black', label='Instability')
ax3.set_xlabel('t', fontsize=12)
ax3.legend(bbox_to_anchor=(0., -0.8, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=9)

ENDINGTIME = datetime.datetime.now().replace(microsecond=0)
TIMEDIFFERENCE = ENDINGTIME - STARTINGTIME
#print 'Computation time:', TIMEDIFFERENCE    
    
