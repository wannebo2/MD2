import numpy as np

def makeInitialConditions(N,dw,types):
    atoms = []
    while len(atoms)<N:
        atoms.append([random.choice(types),np.random(3),(np.random(3)*0.1)-0.05])
def integrateForceOnAtom(atoms,atom):
    global tmstp
    global typedict
    pos = atom[1]
    vel = atom[2]
    for a in atoms:
        if a == atom:
            continue
        params = typedict[atom[0]][a]
        d = np.sum(np.pow(a[1]-pos,2))
        magn = (params[0]*pow(d,-0.5))+(params[1]*pow(d,-1))+(params[2]*pow(d,-1.5))+(params[3]*pow(d,-2))
        dirc = (pos-a[1])/pow(d,0.5)
        vel += dirc*magn*tmstp
    pos += vel*tmstp
    return [atom[0],pos,vel]
def tickSystem(frame):
    for atom in frame:
        atom = integrateForceOnAtom(frame,atom)
    return frame
def makeTypeDict(typlst):
    out = {}
    for t in typlst:
        if not t in out:
            out[t] = {}
        for t2 in typlst:
            if not t == t2:
                if not t2 in out:
                    out[t2] = {}
                out[t][t2] = np.random.random(3)-0.5
                out[t2][t] = out[t][t2]
    return out

    
