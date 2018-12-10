import numpy as np
import transitionMatrix as tm
import valueiteration as vi

def ReadInput(filename):
    obstacles=[]
    with open(filename) as fp:
        s=int(fp.readline())
        o=int(fp.readline())
        for i in range(o):
            obstacles.append(tuple(reversed(list(map(int,fp.readline().strip().split(','))))))
        cars_start=tuple(reversed(list(map(int,fp.readline().strip().split(',')))))
        cars_end=tuple(reversed(list(map(int,fp.readline().strip().split(',')))))
    return s,obstacles,cars_start,cars_end

def Turn_Left(move):
    if move==0:
        return 3
    if move==1:
        return 2
    if move==2:
        return 0
    if move==3:
        return 1
    return -1

def Turn_Right(move):
    if move==0:
        return 2
    if move==1:
        return 3
    if move==2:
        return 1
    if move==3:
        return 0
    return -1

def UpdatePos(size,pos,move):
    if move==0:
        if pos[0]==0:
            return pos
        return pos[0]-1,pos[1]
    if move==1:
        if pos[0]==size-1:
            return pos
        return pos[0]+1,pos[1]
    if move==2:
        if pos[1]==size-1:
            return pos
        return pos[0],pos[1]+1    
    if move==3:
        if pos[1]==0:
            return pos
        return pos[0],pos[1]-1
    return pos

def updateUtility(utilityMatrix, traceMatrix, alpha, delta):
    utilityMatrix+=alpha*delta*traceMatrix
    return utilityMatrix

def updateTraceMatrix(traceMatrix,gamma,Lambda):
    traceMatrix=traceMatrix*gamma*Lambda
    return traceMatrix

def main():
    size,obstacles,car_start,car_end=ReadInput("input.txt")
    
    transitionMatrix=tm.GenerateTransitionMatrix(size)    
    
    rewardMatrix = np.full((size,size),-1.0)
    for a in obstacles:
        rewardMatrix[a]=-101.0
    
    transitionMatrix=tm.GenerateTransitionMatrix(size)
    utilityMatrix,optimalPolicy=vi.GetOptimalPolicy(size,transitionMatrix,rewardMatrix.copy(),car_end)
    
    utility=np.zeros((size,size))
    gamma=0.9
    alpha=0.1
    Lambda=0.5
    
    rewardMatrix[car_end]+=100
    utility[car_end]=rewardMatrix[car_end]
    
    #generates episodes and update online
    for j in range(4000):
        pos = tuple([np.random.randint(0, size), np.random.randint(0, size)])
        #pos=car_start
        dest=car_end
        np.random.seed(j)
        swerve = np.random.random_sample(10000000)
        k=0
        traceMatrix=np.zeros((size,size))
        while pos!=dest:
            move = optimalPolicy[pos]
            if swerve[k] > 0.7:
                if swerve[k] > 0.8:
                    if swerve[k] > 0.9:
                        move = Turn_Left(Turn_Left(move))
                    else:
                        move = Turn_Right(move)
                else:
                    move = Turn_Left(move)
            k+=1
            newpos=UpdatePos(size,pos,move)
            delta=rewardMatrix[pos]+gamma*utility[newpos]-utility[pos]
            traceMatrix[pos]+=1
            utility=updateUtility(utility, traceMatrix,alpha,delta)
            
            traceMatrix=updateTraceMatrix(traceMatrix,gamma,Lambda)
            pos=newpos
    
    print("Dynamic Programming Utility:")
    print(utilityMatrix)
    print("TDlambda Utility:")
    print(utility)
    
if __name__ == "__main__":
    main()