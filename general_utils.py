#assorted vector operations
import math
import numpy
def euclidSqr(a,b):
    amb = np.array(a)-np.array(b)
    return np.dot(amb,amb)
def performRotation(self,A,B,angle):
    newA = (torch.cos(angle)*A)+(torch.sin(angle)*B)
    newB = (torch.cos(angle)*B)-(torch.sin(angle)*A)
    return newA,newB
def getCosine(v,e1,e2): #get cos(theta), where theta is the angle of v projected onto the e1, e2 plane:
    return np.dot(v,e1)/pow(pow(np.dot(v,e1),2)+pow(np.dot(v,e2),2),0.5)
def performRotation3D(self,xv,yv,zv,angleYZ,angleXZ,angleXY):
    yv,zv = performRotation(yv,zv,angleYZ) #First rotate about the x axis
    xv,zv = performRotation(xv,zv,angleXZ) #Then rotate about y
    xv,yv = performRotation(xv,yv,angleXY)
    return xv,yv,zv
def getAngles(v1,v2,v3): #get a set of rotations that transforms normalized vectors v1,v2, and v3 into e1,e2, and e3
    e1 = [1,0,0]
    e2 = [0,1,0]
    e3 = [0,0,1]
    cosv1e1e2 = getCosine(v1,e1,e2)
    cosv1e1e3 = getCosine(v1,e1,e3)
    e1e2 = math.acos(cosv1e1e2)
    e1e3 = math.acos(cosv1e1e3)
    e1,e2 = perform_rotation(e1,e2,-1*e1e2)
    e1,e3 = perform_rotation(e1,e3,-1*e1e3)
    cosv2e2e3 = getCosine(v2,e2,e3)
    e2e3 = math.acos(cosv2e2e3)
    return [e2e3,e1e3,e1e2]
#def applyDenoise(deltas,locations,rotationVectors,denoise_factor):
    #Todo: go through and make sure everything has the right shape
    
