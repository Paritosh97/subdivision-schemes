import matplotlib.pyplot as plt

def plotPolygons(P,scheme):
    datasets = []
    plt.figure()
    
    plt.title(scheme, fontdict=None, loc='center', pad=None)

    for i in range(len(P)):
        poly = P[i].copy()
        poly.append(poly[0]) #repeat the first point to create a 'closed loop'
    
        xs, ys = zip(*poly) #create lists of x and y values

        plt.plot(xs,ys) 
        datasets.append(str(i))
       

    plt.legend(datasets)
    plt.show() 
    
def plotAngles(angles,title): 
    
    plt.title(title, fontdict=None, loc='center', pad=None)

    x = [i for i in range(len(angles))]
    y = angles
    
    plt.scatter(x, y)
    plt.show()

def ChaikinSubdivisionClosedPolygon(N,Polygon): ##int N, Point *P, Point *Q):
    P = Polygon.copy()
    for n in range(N-1):
        Pnew = []
        
        for i in range(len(P[n])-1):
            x1   = 3/4 * P[n][i][0] + 1/4 * P[n][i+1][0]
            y1   = 3/4 * P[n][i][1] + 1/4 * P[n][i+1][1]
            x2   = 1/4 * P[n][i][0] + 3/4 * P[n][i+1][0]
            y2   = 1/4 * P[n][i][1] + 3/4 * P[n][i+1][1]
            
            Pnew.append([x1,y1])  
            Pnew.append([x2,y2])  
            
            #Q[2*i].x = 3./4. * P[i].x + 1./4. * P[i+1].x
            #Q[2*i].y = 3./4. * P[i].y + 1./4. * P[i+1].y
            #Q[2*i+1].x = 1./4. * P[i].x + 3./4. * P[i+1].x
            #Q[2*i+1].y = 1./4. * P[i].y + 3./4. * P[i+1].y
            
        i = len(P[n])-1
        x1   = 3/4 * P[n][i][0] + 1/4 * P[n][0][0]
        y1   = 3/4 * P[n][i][1] + 1/4 * P[n][0][1]
        x2   = 1/4 * P[n][i][0] + 3/4 * P[n][0][0]
        y2   = 1/4 * P[n][i][1] + 3/4 * P[n][0][1]
        
        Pnew.append([x1,y1])  
        Pnew.append([x2,y2]) 
        
        #Q[2*i].x = 3./4. * P[i].x + 1./4. * P[0].x
        #Q[2*i].y = 3./4. * P[i].y + 1./4. * P[0].y
        #Q[2*i+1].x = 1./4. * P[i].x + 3./4. * P[0].x
        #Q[2*i+1].y = 1./4. * P[i].y + 3./4. * P[0].y
        
        P.append(Pnew)

    return P

def CornerCuttingSubdivisionClosedPolygon(A,B,N,Polygon): #double A, double B, int N, Point *P, Point *Q
    P = Polygon.copy()
    for n in range(N-1):
        Pnew = []
        
        for i in range(len(P[n])-1):
            x1   = A * P[n][i][0] + (1-A) * P[n][i+1][0]
            y1   = A * P[n][i][1] + (1-A) * P[n][i+1][1]
            x2   = B * P[n][i][0] + (1-B) * P[n][i+1][0]
            y2   = B * P[n][i][1] + (1-B) * P[n][i+1][1]
            
            Pnew.append([x1,y1])  
            Pnew.append([x2,y2])  
            #Q[2*i].x = A * P[i].x + (1. - A) * P[i+1].x;
            #Q[2*i].y = A * P[i].y + (1. - A) * P[i+1].y;
            #Q[2*i+1].x = B * P[i].x + (1. - B) * P[i+1].x;
            #Q[2*i+1].y = B * P[i].y + (1. - B) * P[i+1].y;
            
        i = len(P[n])-1
        x1   = A * P[n][i][0] + (1-A) * P[n][0][0]
        y1   = A * P[n][i][1] + (1-A) * P[n][0][1]
        x2   = B * P[n][i][0] + (1-B) * P[n][0][0]
        y2   = B * P[n][i][1] + (1-B) * P[n][0][1]
        
        Pnew.append([x1,y1])  
        Pnew.append([x2,y2]) 
        
        #Q[2*i].x = A * P[i].x + (1. - A) * P[0].x;
        #Q[2*i].y = A * P[i].y + (1. - A) * P[0].y;
        #Q[2*i+1].x = B * P[i].x + (1. - B) * P[0].x;
        #Q[2*i+1].y = B * P[i].y + (1. - B) * P[0].y;
        
        P.append(Pnew)

    return P

def getId(i,len):
    n = 0
    if i < 0:
        n=len+i
    elif i > len-1:
        n=i-len
    else:
        n=i
    return n
    
    
    
def GeneralizeFourPointClosedPolygon(N,Polygon, eps):
      
    P = Polygon.copy()
    
    for n in range(N-1):
        Pnew = []
        A    = [0,0]  
        B    = [0,0]  

        for i in range(len(P[n])):

            im1  = getId(i-1,len(P[n]))
            ip1  = getId(i+1,len(P[n]))
            ip2  = getId(i+2,len(P[n]))

            
            A[0] = (P[n][im1][0] + P[n][ip2][0])/2
            A[1] = (P[n][im1][1] + P[n][ip2][1])/2        
            B[0] = (P[n][i][0]   + P[n][ip1][0])/2
            B[1] = (P[n][i][1]   + P[n][ip1][1])/2
        
            x1   = P[n][i][0]
            y1   = P[n][i][1]
            x2   = -eps * A[0] + (1+eps) * B[0]
            y2   = -eps * A[1] + (1+eps) * B[1]
        
            Pnew.append([x1,y1])  
            Pnew.append([x2,y2]) 
        
        
        P.append(Pnew)

    return P

def UniformsplineClosedPolygon(N,Polygon):
    P = Polygon.copy()
    
    for n in range(N-1):
        Pnew = []
        
        for i in range(len(P[n])):
            im2  = getId(i-2,len(P[n]))
            im1  = getId(i-1,len(P[n]))
            ip1  = getId(i+1,len(P[n]))
            
            x1   = 1/8 * (P[n][im1][0] + 6 * P[n][i][0]+ P[n][ip1][0])
            y1   = 1/8 * (P[n][im1][1] + 6 * P[n][i][1]+ P[n][ip1][1])
            x2   = 1/2 * (P[n][i][0] + P[n][ip1][0])
            y2   = 1/2 * (P[n][i][1] + P[n][ip1][1])
        
            Pnew.append([x1,y1])  
            Pnew.append([x2,y2]) 


        P.append(Pnew)
    return P

#Angle functions:

import numpy as np

def theta(a,b,c): 
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def getInnerAngles(Polygon,d): #C2 continuity 
    angles = []
    points = Polygon[d]
    numPoints = len(points)
    

    for i in range(1,numPoints-1):
        angles.append(theta(points[i-1],points[i],points[i+1]))
    
    
    angles.append(theta(points[numPoints-2],points[numPoints-1],points[0]))
    angles.append(theta(points[numPoints-1],points[0],points[1]))

    return(angles)


def getHorizonAngles(Polygon,d): #C1 continuity 
    angles = []
    points = Polygon[d]
    numPoints = len(points)
    
    for i in range(numPoints-1):
        horizon = [0,points[i][1]]
        angles.append(theta(horizon,points[i],points[i+1]))
    
    horizon = [0,points[i][1]]
    angles.append(theta(horizon,points[i+1],points[0]))
    
    return(angles)

#Continuity tests:

def plotC1(Polygon,d):
    plotAngles(getHorizonAngles(Polygon,d),"C1 continuity plot [degree:"+str(d)+"]")
    
def plotC2(Polygon,d):
    plotAngles(getInnerAngles(Polygon,d),"C2 continuity plot [degree:"+str(d)+"]")

#Subdivision Schemes:

def doChaikin(Polygon,N,d):
    scheme = "Chaikin Subdivision Scheme:"
    print(scheme)
    Chaikin  = ChaikinSubdivisionClosedPolygon(N,Polygon)
    plotPolygons(Chaikin,scheme)
    plotC1(Chaikin,d)
    plotC2(Chaikin,d)
    
def doCornerCutting(Polygon,N,A,B,d):
    scheme = "CornerCutting Subdivision Scheme (A:"+str(A)+"; B:"+str(B)+"):"
    print(scheme)
    CorCut_1 = CornerCuttingSubdivisionClosedPolygon( A, B , N, Polygon);
    plotPolygons(CorCut_1,scheme)
    plotC1(CorCut_1,d)
    plotC2(CorCut_1,d)
    
    
def doGenFourPoint(Polygon,N,eps,d):
    scheme = "Generalized Four Point Subdivision Scheme (eps:"+str(eps)+"):"
    print(scheme)
    FourCorners = GeneralizeFourPointClosedPolygon(N,Polygon, eps);
    plotPolygons(FourCorners,scheme)
    plotC1(FourCorners,d)
    plotC2(FourCorners,d)
    
def doUnifSplines3(Polygon,N,d):
    scheme = "Uniform Splines degree 3"
    print(scheme)
    UnifSplines = UniformsplineClosedPolygon(N,Polygon);
    plotPolygons(UnifSplines,scheme)
    plotC1(UnifSplines,d)
    plotC2(UnifSplines,d)

def main():
    
    P0 = [[100,300],[500,300],[500,500],[300,700]]
    
    P0 = [[4,4],[12,4],[12,12],[4,12]]
    Polygon = []
    Polygon.append(P0)
    
    numberOfSteps  = 7 #N degrees
    continuityTest = numberOfSteps -1  #degree of continuity test
    
    
    doChaikin(Polygon,numberOfSteps,continuityTest)
        
    A = 0.52
    B = 0.41
    doCornerCutting(Polygon,numberOfSteps,A,B,continuityTest)

    A = 0.75
    B = 0.25
    doCornerCutting(Polygon,numberOfSteps,A,B,continuityTest)

    
    eps = 1/8
    doGenFourPoint(Polygon,numberOfSteps, eps,continuityTest)
    
    doUnifSplines3(Polygon,numberOfSteps,continuityTest)
    
if __name__ == '__main__':
    main()


