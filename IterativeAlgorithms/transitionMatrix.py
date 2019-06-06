def GenerateTransitionMatrix(size):
    transitionMatrix=dict()
    for x in range(size):
        for y in range(size):
            for a in range(4):
                l=[]
                if a==0:
                    l.append((x-1,y,0.7)) if x>0 else l.append((x,y,0.7))
                    l.append((x+1,y,0.1)) if x<size-1 else l.append((x,y,0.1))
                    l.append((x,y+1,0.1)) if y<size-1 else l.append((x,y,0.1))
                    l.append((x,y-1,0.1)) if y>0 else l.append((x,y,0.1))
                elif a==1:
                    l.append((x-1,y,0.1)) if x>0 else l.append((x,y,0.1))
                    l.append((x+1,y,0.7)) if x<size-1 else l.append((x,y,0.7))
                    l.append((x,y+1,0.1)) if y<size-1 else l.append((x,y,0.1))
                    l.append((x,y-1,0.1)) if y>0 else l.append((x,y,0.1))
                elif a==2:
                    l.append((x-1,y,0.1)) if x>0 else l.append((x,y,0.1))
                    l.append((x+1,y,0.1)) if x<size-1 else l.append((x,y,0.1))
                    l.append((x,y+1,0.7)) if y<size-1 else l.append((x,y,0.7))
                    l.append((x,y-1,0.1)) if y>0 else l.append((x,y,0.1))
                elif a==3:
                    l.append((x-1,y,0.1)) if x>0 else l.append((x,y,0.1))
                    l.append((x+1,y,0.1)) if x<size-1 else l.append((x,y,0.1))
                    l.append((x,y+1,0.1)) if y<size-1 else l.append((x,y,0.1))
                    l.append((x,y-1,0.7)) if y>0 else l.append((x,y,0.7))
                transitionMatrix[x,y,a]=l
    return transitionMatrix