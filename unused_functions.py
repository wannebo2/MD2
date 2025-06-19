# " Fear not, for you are found.
#   You are home... and there is no going back
#   No one leaves this place
#   But what is this place?
#   The answer is [unused_functions.py]
#   It is a collection point for all lost and unloved things. Like you. "
# - Quote from Thor: Ragnarok


def loadDCDres(filename,desiredSteps,pdb,timeConstant = 48.88821,tolerance = 0.0001):
    if not filename.endswith(".pdb"):
        filename += ".pdb"
    if "workingDirectory" in globals():
        global workingDirectory
        if (not workingDirectory in filename):
            filename = workingDirectory+filename
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
def CordToAdress(cds,bound = 1000,neg_bound = 0,dgtLen = 8):
    out = []
    for cd in cds:
        out.append(str((cd+neg_bound)/bound))
        for n in range(10):
            out[-1] = out[-1].replace(str(n),(("Z"*(4-len(str(bin(n)[2:]))))+str(bin(n)[2:])).replace("0","Z").replace("1","O"))
        out[-1] = out[-1][5:]
        if len(out[-1])<dgtLen:
            out[-1] = out[-1]+("Z"*(dgtLen-len(out[-1])))
    return out[5:]

def indexAtom(atm,Tree,bound = 1000,neg_bound = 0,maxItemsPerLv = 10,max_n = 5):
    atmData = {"coord":atm.get_coord(),"name":atm.get_name(),"id":atm.get_id()}
    adr = CordToAdress(atmData["coord"],bound,neg_bound,max_n+2)
    n = 0
    loc = Tree
    while not done:
        Tree["atm_list"].append(atmData)
        if len(Tree["atm_list"]) == maxItems:
            for a in Tree["atm_list"]:
                adr2 = CordToAdress(a["coord"],bound,neg_bound)
                lv2 = "".join([adr2[i][n+1] for i in range(len(adr2))])
                if not lv2 in Tree:
                    Tree[lv2] = {"atm_list":[a]}
                else:
                    Tree[lv2]["atm_list"].append(a)
            break;
        elif len(Tree["atm_list"]) > maxItemsPerLv:
            lv = "".join([adr[i][n] for i in range(len(adr))])
            if not lv in Tree:
                Tree[lv] = {"atm_list":[atmData]}
                break;
            else:
                loc = Tree[lv]
            n += 1
        if n>max_n:
            break;
            
def makeAtomTree(atms,bound,neg_bound,maxItemsPerLv):
    Tree = {"atm_list":[]}
    for a in atms:
        indexAtm(a,Tree,bound,neg_bound,maxItemsPerLv)
    return Tree

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
    posembedding = makeEmbedding(coords,torch.flatten([normalVec,unitVec]),scales,rotscales)
    return {"resname":name,"ssegid":ID,"direction":unitVec,"rotation":normalVec,"coords":coords,"posembedding":posembedding}
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
