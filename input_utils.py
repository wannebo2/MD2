# todo: everything
import Bio.PDB

def loadPDB(filename,ID="0"):
    global workingdirectory
    if not filename.endswith(".pdb"):
        filename += ".pdb"
    if not filename.beginswith(workingdirectory):
        filename = workingdirectory+filename
    z = Bio.PDB.PDBParser.PDBParser()
    out = z.get_structure(ID,filename)
    del z
    return out
