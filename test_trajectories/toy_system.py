import numpy as np
import random
import MDAnalysis
import matplotlib.pyplot as plt
import copy
def integrateForceOnAtom(atoms,atom):
    global tmstp
    global typedict
    global bondDist
    global bondCoeff
    global MAX
    pos = atom[1]
    vel = atom[2]
    bondlist = atom[3]
    for c in range(len(atoms)):
        a = atoms[c]
        if (a[1] == atom[1]).all():
            continue
        params = typedict[atom[0]][a[0]]
        d = np.sum(np.pow(a[1]-pos,2))
        dirc = (pos-a[1])/pow(d+0.001,0.5)
        if c in bondlist:
            magnA = bondCoeff*(bondDist-pow(d,0.5))
            magnB = 0
        else:
            magnB = (params[1]*pow(d,-1))+(params[2]*pow(d,-1.5))+(params[3]*pow(d,-2))
            magnA = 0
        if abs(magnB)>MAX:
            #print(magnB)
            magnB = MAX*(magnB/abs(magnB))
           # print("erm")
        vel += dirc*magnB*tmstp
        #if random.random()<0.1:
        vel += (dirc*magnA)*tmstp
    pos += (vel*tmstp)#+(dirc*magnA)
    return [atom[0],pos,vel,atom[3]]
def tickSystem(frame):
    #random.shuffle(frame)
    out = []
    xs = []
    ys = []
    zs = []
    #frame = 
    for atom in frame:
        out.append(integrateForceOnAtom(frame,atom))
        xs += [out[-1][1][0]]
        ys += [out[-1][1][1]]
        zs += [out[-1][1][2]]
    return out,xs,ys,zs
def makeTypeDict(typlst):
    out = {}
    for t in typlst:
        if not t in out:
            out[t] = {}
        for t2 in typlst:
            if True:
                if not t2 in out:
                    out[t2] = {}
                k = (np.random.random(4)-0.5)*1
                out[t][t2] = k
                out[t2][t] = k
    return out
def makeBonds(frame,bindWithin):
    bonds = []
    for c in range(len(frame)):
        for c2 in range(c):
            atom = frame[c]
            atom2 = frame[c2]
            if not c == c2:
                d = pow(np.sum(pow(atom[1]-atom2[1],2)),0.5)
                if random.random()<0.1 or (d<bindWithin):
                    atom[3].append(c2)
                    atom2[3].append(c)
                    bonds += [[c,c2]]
    return bonds
def makeInitialState(N,bound = 4,vbound = 0):
    global typedict
    frame = []
    while len(frame)<N:
        frame.append([random.choice(list(typedict.keys())),bound*np.random.normal(size=3),vbound*(np.random.normal(size=3)-0.5),[]])
    return frame
def makePSF(bonds,filename):
    toWrite = ["!NBONDS",""]
    c = 0
    while c<len(bonds):
        toWrite[-1] += str(bonds[c][0])+" "+str(bonds[c][1])+" "
        if c%2 == 1:
            toWrite += [""]
        c += 1
    f = open(filename, mode = 'w')
    f.write("/n".join(toWrite))
    f.close()
tmstp = 0.01
bondDist = 2
bondCoeff = 1
bindWithin = 1
typlst = ["A","B","C","D"]
N = 20
trajN = 1000
trajL = 10
MAX = 1000
sT = 1
typedict = makeTypeDict(typlst)
#print(typedict)
trajs = []
bonds = []
inits = []
save_path = "test_trajectories"
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
while len(trajs)<trajN:
    frames = []
    t = 0
    inits.append(makeInitialState(N))
    bonds.append(makeBonds(inits[-1],bindWithin))
    frames.append(inits[-1])
    while t<trajL:
        newframe,xs,ys,zs = tickSystem(frames[-1])
        frames.append(copy.deepcopy(newframe))
        t += tmstp
        if round(t,0)==round(t,4):
            print(str(round(100*t/trajL,1))+"% complete with trajectory number "+str(len(trajs)))
            ax.clear()
            ax.scatter(xs,ys,zs)
            ax.set_xbound(-20,20)
            ax.set_ybound(-20,20)
            ax.set_zbound(-20,20)
            fig.canvas.draw()
            fig.canvas.flush_events()
    trajs.append(frames)
    print(len(trajs))
    frame_coordinates = []
    frame_vels = []
    uni = MDAnalysis.Universe.empty(N,trajectory = True,velocities=True)
    nms = []
    for a in frames[0]:
        nms.append(a[0])
    uni.add_TopologyAttr('name',nms)
    uni.add_TopologyAttr('bonds',bonds[-1])
    for f in frames:
        #print(f[0][1])
        frame_coordinates.append([])
        frame_vels.append([])
        for a in f:
            frame_coordinates[-1].append(a[1])
            frame_vels[-1].append(a[2])
    uni.load_new(np.array(frame_coordinates))
    #uni.atoms.positions = np.array(frame_coordinates)
    for c in range(len(uni.trajectory)):
        ts = uni.trajectory[c]
        ts.has_velocities = True
        ts.velocities = np.array(frame_vels[c])
    uni.atoms.write(save_path+str(len(trajs))+".pdb")
    #makePSF(bonds[-1],save_path+str(len(trajs))+".psf")
    print("saving to file:")
    with MDAnalysis.Writer(save_path+str(len(trajs))+".dcd",N) as writer:
        for ts in uni.trajectory:
            #uni.atoms.positions = np.array(frame_coordinates[c])
            #uni.atoms.velocities = np.array(frame_vels[c])
            writer.write(uni)
    print("saved.")
    

    
