import torch
import MDAnalysis
import numpy as np
def get_traj(locs,N):
    #print(locs.shape)
    frames = torch.split(locs,N,dim=0)
    #print(frames.shape)
    vels = []
    coords = []
    tms = []
    for loc in frames:
        tms.append(loc[0,-1])
        coords.append(torch.stack([loc[:,0],loc[:,2],loc[:,4]],dim=-1))
        vels.append(torch.stack([loc[:,1],loc[:,3],loc[:,5]],dim=-1))
    return coords,vels,tms
def save_trajectory(locs,N,pdb,filename):
    locs = locs.cpu().detach()
    print(pdb)
    uni = MDAnalysis.Universe(pdb,trajectory = True,velocities=True)
    coords,vels,tms = get_traj(locs,N)
    print(np.array(coords).shape)
    uni.load_new(np.array(coords))
    for c in range(len(uni.trajectory)):
        ts = uni.trajectory[c]
        ts.has_velocities = True
        ts.velocities = np.array(vels[c])
        ts.time = tms[c]
    with MDAnalysis.Writer(filename+".dcd",len(uni.atoms)) as writer:
        for ts in uni.trajectory:
            writer.write(uni)
    print("trajectory saved to "+filename+".dcd")
    
