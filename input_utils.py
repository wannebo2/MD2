# todo: basically everything
import Bio.PDB
import general_utils

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

def getResidueInfo(residue):
    name = residue.get_resname()
    ID = residue.get_segid()
    residue.sort()
    atms = residue.get_atoms()
    o1 = atms[0]
    o2 = atms[1]
    if "orientationAtomsByResidue" in globals():
        global orientationAtoms
        if name in orientationAtomsByResidue:
            for z in atms:
                if z.get_name() == orientationAtomsByResidue[name]["o1"]:
                    o1 = z
                if z.get_name() == orientationAtomsByResidue[name]["o2"]:
                    o2 = z
    c1 = o1.get_coord()
    c2 = o2.get_coord()
    unitVec = general_utils.normDisp(c1,c2)
    coords = general_utils.avg([c1,c2])
    return {"resname":name,"ssegid":ID,"direction":unitVec,"coords":coords}

    
    
