#break ties: N,S,E,W

import numpy as np
import sys        
    
def CalculateUtility(x,y,transitionMatrix,u,rewardMatrix):
    ma=-sys.maxsize-1
    action=0
    for a in range(4):
        l=transitionMatrix[x,y,a]
        v=0
        for s in l:
            v+=u[s[0],s[1]]*s[2]
        if v>ma:
            ma=v
            action=a
    return rewardMatrix[x,y]+0.9*ma,action
    
def GetOptimalPolicy(size,transitionMatrix,rewardMatrix,dest):
    gamma=0.9
    epsilon=0.1
    cm=epsilon*(1-gamma)/gamma
    rewardMatrix[dest]+=100
    u=np.zeros((size,size))
    u1=np.zeros((size,size))
    policy=np.full((size,size),-1)
    while True:
        u=u1.copy()
        delta = 0
        for x in range(size):
            for y in range(size):
                if (x,y)==dest:
                    u1[x,y],policy[x,y]=rewardMatrix[x,y],-1
                else:
                    u1[x,y],policy[x,y]=CalculateUtility(x,y,transitionMatrix,u,rewardMatrix)
                delta=max(delta,np.abs(u1[x,y]-u[x,y]))
        if delta<cm:
            rewardMatrix[dest]-=100
            return u,policy