import Bio.PDB
from MDAnalysis.lib.formats.libdcd import DCDFile
import MDAnalysis
import general_utils
import torch
import copy
import json
import numpy as np
import random
import os
def loadPDB(filename,ID="0"):
    if not filename.endswith(".pdb"):
        filename += ".pdb"
    if "workingDirectory" in globals():
        global workingDirectory
        if (not workingDirectory in filename):
            filename = workingDirectory+filename
    z = Bio.PDB.PDBParser()
    out = z.get_structure(ID,filename)
    del z
    return out

def loadPSF(filename):
    if "workingDirectory" in globals():
        global workingDirectory
        if (not workingDirectory in filename):
            filename = workingDirectory+filename
    f = open(filename,mode = "r")
    z = f.readlines()
    Bonds = {}
    reading = False
    for line in z:
        if "!" in line and reading:
            break;
        elif reading:
            ns = line.split()
            c = 0
            while(c<len(ns)):
                if not ns[c] in Bonds:
                    Bonds[ns[c]] = []
                if not ns[c+1] in Bonds:
                    Bonds[ns[c+1]] = []
                Bonds[ns[c]].append(ns[c+1])
                Bonds[ns[c+1]].append(ns[c])
                c += 2
        elif "!NBOND" in line:
            reading = True
    return Bonds

def getDCDlength(filename,timeConstant = 48.88821):
    #if "workingDirectory" in globals():
     #if (not filename.beginswith(workingDirectory)):
     #   filename = workingdirectory+filename
    z = DCDFile(filename)
    return len(z)*z.header['delta']*z.header['nsavc']*timeConstant/1000
def loadDCD(filename,desiredSteps,pdb,bonds,timeConstant = 48.88821,tolerance = 0.0001):
    #takes in dcd file and a list of times to take data from, and returns a list of atom embeddings, along with a list of coordinates, velocities, times, and rotation vectors
    global AtomWeights
    global AtomEmbeddings
    #global tempoScales
    #global posScales
    global AtomEmbeddingSize
    if not filename.endswith(".pdb"):
        filename += ".pdb"
    if "workingDirectory" in globals():
        global workingDirectory
        if (not filename.beginswith(workingDirectory)):
            filename = workingdirectory+filename
    z = DCDFile(filename)
    timestep = z.header['delta']*z.header['nsavc']*timeConstant
    frame0 = z.tell()
    DCDtoPDBmap = {}
    PDBtoDCDmap = {}
    PDBsById = {}
    atmList = []
    print("matching atoms... (sorry this function might be slow)")
    for atm in pdb.get_atoms():
        for c in range(len(frame0.x)):
            if general_utils.euclidSqr(frame0.x[c],atm)<tolerance:
                DCDtoPDBmap[c] = atm.get_id()
                PDBtoDCDmap[atm.get_id()] = c
                break;
        if not atm in PDBtoDCDmap:
            print("Error: atom could not be found in DCD file. "+atm.get_name()+" "+atm.get_id())
        else:
            PDBsById[atm.get_id()] = atm
            atmList.append(atm)
    print("loading selected frames...")
    aEmbeds = []
    locs = []
    for s in desiredSteps:
        z.seek(round(s/timestep))
        frame = z.tell()
        z.seek(round(s/timestep)+1)
        nxtFrame = z.tell()
        for atm in atmList:
            coords = frame.x[PDBtoDCDmap[atm.get_id()]]
            velocs = (nxtFrame.x[PDBtoDCDmap[atm.get_id()]]-coords)/timestep
            for bondedAtm in bonds[atm.get_id()]:
                coord2 = frame.x[PDBtoDCDmap[bondedAtm]]
                rotvec += (coords-coord2)
                if not PDBsById[bondedAtm].get_name() in AtomWeights:
                    if PDBsById[bondedAtm].get_name()[0] in AtomWeights:
                        AtomWeights[PDBsById[bondedAtm].get_name()] = AtomWeights[PDBsById[bondedAtm].get_name()[0]]
                    else:
                        AtomWeights[PDBsById[bondedAtm].get_name()] = 1
                rotvec2 += AtomWeights[PDBsById[bondedAtm].get_name()]*(coords-coord2) #rot2 is for the computing a normal vector for the rotation (from a weighted sum of bond directions), because we'll need that too.
            rotvec /= np.dot(rotvec,rotvec)+0.00001
            rotvec2 -= np.dot(rotvec2,rotvec)*rotvec
            rotvec2 /= np.dot(rotvec2,rotvec2)+0.00001
            rotvec3 = np.ones(rotvec.shape)
            rotvec3 -= np.dot(rotvec3,rotvec)*rotvec
            rotvec3 -= np.dot(rotvec3,rotvec2)*rotvec2
            rotvecs = [rotvec,rotvec2,rotvec3]
            #posEmbed = makeEmbedding([s,coords,rotvec,rotvec2,rotvec3],[tempoScales,posScales,rotScales,rotScales,rotScales])
            loc = [coords[0],velocs[0],coords[1],velocs[1],coords[2],velocs[2],s]
            if not atm.get_name() in AtomEmbeddings:
                if atm.get_name()[0] in AtomEmbeddings:
                    AtomEmbeddings[atm.get_name()] = AtomEmbeddings[atm.get_name()[0]]
                    print("No embedding found for "+str(atm.get_name()))
                    print("using the existing embedding for "+str(atm.get_name()[0]))
                else:
                    print("No embedding found for "+str(atm.get_name()))
                    print("using a random one.")
                    AtomEmbeddings[atm.get_name()] = np.random.random(AtomEmbeddingSize)
            atmEmbed = AtomEmbeddings[atm.get_name()]
            aEmbeds.append(atmEmbed)
            locs.append(loc)
            rots.append(rotvecs)

    z.close()
    return aEmbeds,locs,rots #atom embeddings is of shape (N,d), where N is the number of atoms. locs is of shape (N.7), and holds coordinate and velocity information. Rots is of shape (N,(3,3)), and holds rotation matricies

def load_trajectory(pdb,dcd,desiredSteps):
    global AtomWeights
    global AtomEmbeddings
    global AtomEmbeddingSize
    uni = MDAnalysis.Universe(pdb,dcd)
    ts = uni.trajectory[0]
    atoms = uni.atoms
    dt = ts.dt
    aEmbeds = []
    locs = []
    rots = []
    for s in desiredSteps:
        #print(s)
        fr = round(s/dt)
        if fr<0:
            print("there is an issue somewhere.")
        fr = max([fr,1])
        fr = min([fr,len(uni.trajectory)-1])
        lts = np.array(uni.trajectory[fr-1].positions)
        ts = uni.trajectory[fr]
        for c in range(len(atoms)):
            pos = ts.positions[c]
            vel = (pos-lts[c])/dt#ts.velocities[c]
            #print(vel)
            bonded = atoms[c].bonded_atoms
            nm = atoms[c].name
            if not nm in AtomEmbeddings:
                AtomEmbeddings[nm] = list(np.zeros(AtomEmbeddingSize))
                AtomEmbeddings[nm][len(list(AtomEmbeddings.keys()))] = 1
                f = open("AtomEmbeddings.json",mode='w')
                json.dump(AtomEmbeddings,f)
                f.close()
            embed = AtomEmbeddings[nm]
            rotvec = torch.zeros(3)
            rotvec[-1] = 0.001
            rotvec2 = 0.001*torch.ones(3)
            rotvec2[0] = -0.001
            rotvec3 = torch.ones(3)
            for b in bonded:
                p = ts.positions[b.ix-1]
                rotvec += p-pos
                if not b.name in AtomWeights:
                    AtomWeights[b.name] = random.random()
                    f = open("AtomWeights.json",mode='w')
                    json.dump(AtomWeights,f)
                    f.close()
                rotvec2 += (AtomWeights[b.name]+0.1*random.random())*(p-pos)
            rotvec /= pow(np.dot(rotvec,rotvec),0.5)#+0.001
            rotvec2 -= np.dot(rotvec2,rotvec)*rotvec
            rotvec2 /= pow(np.dot(rotvec2,rotvec2),0.5)#+0.001
            rotvec3 -= np.dot(rotvec3,rotvec)*rotvec
            rotvec3 -= np.dot(rotvec3,rotvec2)*rotvec2
            rotvec3 /= pow(np.dot(rotvec3,rotvec3),0.5)#+0.001
            #print(rotvec)
            #print(rotvec2)
            #print(rotvec3)
            loc = [pos[0],vel[0],pos[1],vel[1],pos[2],vel[2],s]
            rotvecs = torch.transpose(torch.stack([rotvec,rotvec2,rotvec3]),-2,-1)
            aEmbeds.append(embed)
            locs.append(loc)
            rots.append(rotvecs)
    return aEmbeds,locs,torch.stack(rots)



#test function
workingDirectory = os.getcwd()+"\\"
AtomEmbeddingSize = 2240
NeedToLoad = ["AtomWeights","AtomEmbeddings"]#,"EmbedScales"]

setglob = lambda thing,value: exec(thing+" = "+str(value),globals())
for z in NeedToLoad:
    try:
        f = open(z+".json",mode='r')
        setglob(z,json.loads(f.read()))
        f.close()
    except:
        print(z+" could not be loaded.")
        setglob(z,{})
#ScaleList = ["posScales","tempoScales","rotScales"]
#for z in ScaleList:
#    if not z in EmbedScales:
#        EmbedScales[z] = [i for i in range(10)]
#    setglob(z,EmbedScales[z])
