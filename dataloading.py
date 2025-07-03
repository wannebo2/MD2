import os
import torch
from torch.utils.data import Dataset
import input_utils
import numpy as np
class TrajectoryDataset(Dataset):
    def __init__(self,directory):
        self.dir = directory
        self.trajNames = []
        self.trajLens = []
        for thing in os.listdir(self.dir):
            if thing[-4:] == ".dcd":
                self.trajNames  += [self.dir+thing]
                self.trajLens += [input_utils.getDCDlength(self.trajNames[-1])]
        while len(self.trajNames)<15:
            print("too few trajectories. Using random data.")
            self.trajNames += ["RANDOM"]
            self.trajLens += [-1]
            
        self.FrameSD = 100 #standard deviation of sample times chosen, in femetoseconds
        self.FrameNumberMean = 3.0
        self.FrameNumberSD = 1.0
        self.NoiseLevel = 0.05 #Noise level, units tbd
        self.NoiseScales = [1,0.1,1,0.1,1,0.1,0]
        self.RotNoiseScale = 0.1
        self.NoisePerFemetoSecond = 0.001 #Noise increase per femetosecond from first sampled frame
        
    def __len__(self):
        return len(self.trajNames)
    def __getitem__(self,number): #loads some random frames from trajectory number number
        f = self.trajNames[number]
        if f == "RANDOM": #For debugging purposes
            d = 1000
            N = 200
            atmEmbeds = torch.rand((N,d))
            noised = torch.rand((N,7))
            noisedRots = torch.rand((N,3,3))
            svd = torch.linalg.svd(noisedRots)
            noisedRots = svd[0] @ svd[2]
            locs = torch.ones((N,7))
            rots = torch.ones((N,3,3))
            return [atmEmbeds,noised,noisedRots],locs,rots
        pdb = input_utils.loadPDB(f)
        psf = input_utils.loadPSF(f)
        l = self.trajLens[number]
        framesToLoad = max([2,round(np.random.normal(loc=self.FrameNumberMean,scale=self.FrameNumberSD))])
        frameMean = max([min([round(np.random.random()*l),l-self.FrameSD]),self.FrameSD])
        steps = []
        noise = []
        while len(steps)<framesToLoad:
            rndNumber = round(np.random.normal(loc=frameMean,scale=self.FrameSD))
            if not rndNumber in s:
                steps.append(rndNumber)
        mnTime = min(steps)
        atmEmbeds,locs,rots = input_utils.loadDCD(f,steps,pdb,psf,timeConstant = 48.88821,tolerance = 0.0001)
        LocLen = len(locs[0])
        noised = []
        for atm in range(len(atmEmbeds)):
            time = locs[atm][3]
            noised.append(locs[atm] + (((time-mnTime)*self.NoisePerFemetoSecond)+self.NoiseLevel)*torch.normal(torch.zeros(LocLen),self.NoiseScales)) #add an amount of noise to each atom's postion, rotation, velocity, ect., propotional to the time since the first frame. Each location entry can have a different noise standard dev, determined by NoiseScales.
            noisedRots.append(rots[atm] + (((time-mnTime)*self.NoisePerFemetoSecond)+self.NoiseLevel)*torch.normal(torch.zeros(rots.shape),self.RotNoiseScale))
        return [atmEmbeds,noised,noisedRots],locs,rots #not sure if this is actually the right format for this, will probably have to come back and fix
        
