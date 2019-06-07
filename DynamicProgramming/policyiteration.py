#break ties: N,S,E,W

import numpy as np
import sys
    
def PolicyEvaluation(transitionMatrix,u,rewardMatrix,policy,size):
    for x in range(size):
        for y in range(size):
            if rewardMatrix[x,y]==99:
                u[x,y]=99
                continue
            v=0
            l=transitionMatrix[x,y,policy[x,y]]
            
            for s in l:
                v+=u[s[0],s[1]]*s[2]
            u[x,y]=rewardMatrix[x,y]+0.9*v

def GetExpectedAction(x,y,u,T):
    ma=-sys.maxsize - 1
    action=0
    for a in range(4):
        l=T[x,y,a]
        v=0
        for s in l:
            v+=u[s[0],s[1]]*s[2]
        if v>ma:
            ma=v
            action=a
    return action

def GetOptimalPolicy(size,transitionMatrix,rewardMatrix,dest):
    gamma=0.9
    epsilon=0.1
    com=epsilon*(1-gamma)/gamma
    rewardMatrix[dest]=99
    u=np.zeros((size,size))
    u1=np.zeros((size,size))
    policy=np.zeros((size,size),int)
    policy[dest]=-1
    while True:
        u=u1.copy()
        PolicyEvaluation(transitionMatrix,u1,rewardMatrix,policy,size)
        delta = np.absolute(u1-u).max()
        unchanged=True
        if delta<com:
            return u, policy
        for x in range(size):
            for y in range(size):
                if rewardMatrix[x,y]==99:
                    continue
                a=GetExpectedAction(x,y,u1,transitionMatrix)
                if a!=policy[x,y]:
                    policy[x,y]=a
                    unchanged=False
        if unchanged==True:
            return u,policy
        
