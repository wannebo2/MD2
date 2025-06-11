# todo: basically everything
# Important:
#  - figure out how to load and decode DCD files
#  - test input functions
#  - add input/output visualization
#  - figure out what the model is going to be and implement it in pytorch
#  - figure out how to talk to vmd/namd2 to automatically run the simulations that would be most useful to train on
# Later:
#  - make spatial tree more efficient (lower number base)
import Bio.PDB
from MDAnalysis.lib.formats.libdcd import DCDFile
import general_utils
import copy
def loadPDB(filename,ID="0"):
    if not filename.endswith(".pdb"):
        filename += ".pdb"
    if "workingDirectory" in globals():
        global workingDirectory
        if (not filename.beginswith(workingDirectory)):
            filename = workingdirectory+filename
    z = Bio.PDB.PDBParser.PDBParser()
    out = z.get_structure(ID,filename)
    del z
    return out

def loadDCD(filename,desiredSteps,pdb,timeConstant = 48.88821,tolerance = 0.0001):
    if not filename.endswith(".pdb"):
        filename += ".pdb"
    if "workingDirectory" in globals():
        global workingDirectory
        if (not filename.beginswith(workingDirectory)):
            filename = workingdirectory+filename
    z = DCDFile(filename)
    timestep = z.header['delta']*z.header['nsavc']*timeConstant
    out = []
    frame0 = z.tell()
    rlist = []
    for res in pdb.get_residues():
        r = getResidueInfo(res)
        for c in range(len(frame0.x)):
            if sum(abs(frame0.x[c]-r["coords"]))<tolerance:
                r["assigned atom"] = c
                break;
        if not "assigned atom" in r:
            print("Error: atom could not be matched to residue "+r["resname"]+" "+r["ssegid"]+"!")
        else:
            rlist.append(r)
    for s in desiredSteps:
        z.seek(round(s/timestep))
        frame = z.tell()
        rcops = []
        for r in rlist:
            rcop = copy.copy(r)
            rcop["coords"] = frame.x[r["assigned atom"]]
            rcops.append(rcop)
        out += [{"time":s,"coord tree":buildSpatialTree(rcops),"residue list":rcops}]
    z.close()
    return out

def getResidueInfo(residue):
    name = residue.get_resname()
    ID = residue.get_segid()
    residue.sort()
    atms = residue.get_atoms()
    o1 = atms[0]
    o2 = atms[0]
    o3 = atms[0]
    if len(atms)>1:
        o2 = atms[1]
        o3 = atms[1]
        if len(atms)>2:
            o3 = atms[2]
    if "orientationAtomsByResidue" in globals():
        global orientationAtoms
        if name in orientationAtomsByResidue:
            for z in atms:
                if z.get_name() == orientationAtomsByResidue[name]["o1"]:
                    o1 = z
                if z.get_name() == orientationAtomsByResidue[name]["o2"]:
                    o2 = z
                if z.get_name() == orientationAtomsByResidue[name]["o3"]:
                    o3 = z
    c1 = np.array(o1.get_coord())
    c2 = np.array(o2.get_coord())
    c3 = np.array(o3.get_coord())
    unitVec = (c2-c1)#general_utils.normDisp(c1,c2)
    unitVec = unitVec/(np.dot(unitVec,unitVec)+0.0001)
    normalVec = c3-c1
    normalVec = normalVec-(np.dot(unitVec,normalVec)*unitVec)
    normlVec /= (np.dot(normalVec,normalVec)+0.0001)
    coords = c1#general_utils.avg([c1,c2])
    return {"resname":name,"ssegid":ID,"direction":unitVec,"rotation":normalVec,"coords":coords}

def buildSpatialTree(residueList, maxlevels = 5, maxResPerLv = 5,strtDgt = -3): #make a tree structure containing all the residues.
    maxlevels = maxlevels + strtDgt
    if residueIgnoreList in globals():
        residueIgnoreList = globals()["residueIgnoreList"]
    else:
        residueIgnoreList = []
    if residueAcceptList in globals():
        residueAcceptList = globals()["residueAcceptList"]
    else:
        residueAcceptList = []
    Tree = {}
    for res in residueList:
        if res["resname"] in residueIgnoreList:
            pass
        elif (len(residueAcceptList)>0) and not res["resname"] in residueAcceptList:
            pass
        else:
            loc = Tree
            dgt = strtDgt
            while dgt<maxlevels: #shouldn't reach maxlevels most of the time, but just in case two residues have the same coordinates, maxlevels shouldn't be excessivley high
                dgts = "#".join([str(round(r["coords"][c],dgt)) for c in len(r["coords"])])
                if not dgts in loc:
                    loc[dgts] = {"residueList":[r]}
                    break;
                elif len(loc[dgts]["residueList"]) > maxResPerLv:
                    loc[dgts]["residueList"] += [r]
                    loc = loc[dgts]
                elif len(loc[dgts]["residueList"]) < maxResPerLv:
                    loc[dgts]["residueList"] += [r]
                    break;
                else:
                    loc[dgts]["residueList"] += [r]
                    for r2 in loc[dgts]["residueList"]:
                        dgts2 = "#".join([str(round(r["coords"][c],dgt+1)) for c in len(r["coords"])])
                        if dgts2 in loc[dgts]:
                            loc[dgts][dgts2]["residueList"] += [r2]
                        else:
                            loc[dgts][dgts2] = {"residueList":[r2]}
                    break;
                dgt += 1
                if dgt == maxlevels:
                    dgts = "#".join([str(round(r["coords"][c],dgt)) for c in len(r["coords"])])
                    if dgts in loc:
                        loc[dgts]["residueList"] += [r]
                    else:
                        loc[dgts] = {"residueList":[r]}
    return Tree
def getItemsWithinRadius(Tree,coord,radius): #get a list of residues within a given radius from a point
    # currently this is not very efficient for randomly placed items- TODO: make the tree created by the above function and used by this function using binary coordinates instead of decimal ones.
    rad2 = pow(radius,2)
    out = []
    if len(list(Tree.keys()))<2:
        for res in Tree["residueList"]:
            if general_utils.euclidSqr(res["coords"],coord)<rad2:
                out += [res]
        return out
    for z in Tree:
        if "#" in z:
            dgts = z.split("#")
            maxdispcoord = []
            mindispcoord = []
            for d in dgts:
                f = float(dgts)
                if f-coord[c]>0:
                    maxdispcoord.append(f+0.5)
                    mindispcoord.append(f-0.5)
                else:
                    maxdispcoord.append(f-0.5)
                    mindispcoord.append(f+0.5)
            if general_utils.euclidSqr(mindispcoord,coord)<radius:
                if general_utils.euclidSqr(maxdispcoord,coord)<radius:
                    out += Tree[z]["residueList"]
                else:
                    out += getItemsWithinRadius(Tree[z],coord,radius)


