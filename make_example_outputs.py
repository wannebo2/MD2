import torch
import models
import dataloading
import output_utils
import loss_functions
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random
import time
import math
def make_trajectories(data_loader,model):
    global device
    for i,data in enumerate(data_loader):
        stoof,expLocs,expRots,pdb = data
        atmEmbeds = stoof[0]
        noisedLocs = stoof[1]
        noisedRots = stoof[2]
        atmEmbeds = atmEmbeds.to(device)
        noisedLocs = noisedLocs.to(device)
        noisedRots = noisedRots.to(device)
        expLocs = expLocs.to(device)
        expRots = expRots.to(device)
        atms = atmEmbeds.shape[1]
        outLocs,outRots = model(atmEmbeds,noisedLocs,noisedRots)
        for c in range(len(outLocs)):
            pdbSpl = pdb[c].split("/")
            b = outLocs[c]
            output_utils.save_trajectory(b,20,pdb[c],"outputs/O"+str(pdbSpl[-1][:-4]))
        for c in range(len(noisedLocs)):
            pdbSpl = pdb[c].split("/")
            b = noisedLocs[c]
            output_utils.save_trajectory(b,20,pdb[c],"outputs/N"+str(pdbSpl[-1][:-4]))
        for c in range(len(expLocs)):
            pdbSpl = pdb[c].split("/")
            b = expLocs[c]
            output_utils.save_trajectory(b,20,pdb[c],"outputs/E"+str(pdbSpl[-1][:-4]))
        #optimizer.zero_grad()
       # print("cycle")
        #I += 1
        #if i % 2 == 0:
          #  print(str(round(i/len(data_loader),3)*100)+" % complete with epoch number "+str(c))
           # print(losses[-1])
    print("done!")
path = "test_trajectories/"
save_path = "model_saves/"
print("loading dataset...")
dataset = dataloading.TrajectoryDataset(path)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True)
print("initializing model...")
load = True
if load:
    j = input("What is the name of the file you would like to load?")
    model = torch.load(save_path+j,weights_only=False)
else:
    model = models.DebuggingModel(6)
params = sum(p.numel() for p in model.parameters())
prefixes = ["","K","M","B","T"]
n = 0
while pow(10,n*3)<params:
    n += 1
print(str(round(params/pow(10,3*(n-1)),2))+prefixes[n-1]+" parameter model loaded.")
if torch.cuda.is_available:
    device = torch.device('cuda')
    print("runing on CUDA")
else:
    device = torch.device('cpu')
    print("running on CPU")
model.to(device)
make_trajectories(data_loader,model)

