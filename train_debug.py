import torch
import models
import dataloading
import loss_functions
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from galore_torch import GaLoreAdamW8bit
import random
import time
import math
def train(data_loader,model,c):
    global device
    losses = []
    I = 0
    #Kl = random.randint(0,199)
    #Kl2 = random.randint(0,6)
    #print(Kl2)
    #Kr2 = random.randint(0,2)
    #Kr3 = random.randint(0,2)
    stEp = time.time()
    a1s = []
    a2s = []
    baseline_losses = []
    for i,data in enumerate(data_loader):
        c1 = time.time()
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
        #print(atms)
        #print(noisedRots.shape)
        #print(noisedLocs.shape)
        #print(atmEmbeds.shape)
        #print("a")
        c2 = time.time()
        a1s.append(c2-c1)
        outLocs,outRots = model(atmEmbeds,noisedLocs,noisedRots)
        #print(expLocs)
        #print(outLocs)
        #print(expRots)
        #print(outRots)
        #print("location rnd value: "+str(outLocs[0,Kl,Kl2]))
        #print("rotation rnd value: "+str(outRots[0,Kr2,Kr3]))
        #print("b")
        loss = loss_functions.compute_loss(outLocs,outRots,expLocs,expRots,noisedLocs,noisedRots)#,outDisp)
        #baseline_loss = loss_functions.compute_loss(noisedLocs,noisedRots,expLocs,expRots,outDisp)
        #print(outLocs.shape)
        #print(outRots.shape)
        #print(loss)
        losses.append((torch.mean(loss)).cpu().detach()) #/(atms*(6+9))
        #baseline_losses.append((torch.mean(baseline_loss)).cpu().detach())
        #print("c")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        c3 = time.time()
        a2s.append(c3-c2)
        #optimizer.zero_grad()
       # print("cycle")
        I += 1
        #if i % 2 == 0:
          #  print(str(round(i/len(data_loader),3)*100)+" % complete with epoch number "+str(c))
           # print(losses[-1])
    endEp = time.time()
    Prop = sum(a1s)/sum(a2s)
    return torch.mean(torch.tensor(losses)),0,endEp-stEp,Prop
    print("done!")
path = "test_trajectories/"
save_path = "model_saves/"
print("loading dataset...")
dataset = dataloading.TrajectoryDataset(path)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True)
print("initializing model...")
load = False
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
print("initializing optimizer...")
param_groups = [{'params':[]},{'params':model.parameters(),'rank':128,'update_proj_gap':200,'scale':0.25,'proj_type':'std'}]
optimizer = GaLoreAdamW8bit(param_groups,lr=0.01)
l = len(data_loader)
N = 1000
print("training for "+str(N)+" epochs")
c = 0
plt.ion()
fig, axs = plt.subplots(1, 1)
axs.set_xlabel("Epoch")
axs.set_ylabel("Logarthim of Loss")
axs.grid(True)
#axs.yaxis.set_minor_locator(MultipleLocator(10))
#axs.xaxis.set_minor_locator(MultipleLocator(10))
Alosses = []
losses = []
epochs = []
baseLosses = []
Aepochs = []
A = 3
while c<N:
    los, baselos,eptime,timeprop = train(data_loader,model,c)
    print("Epoch "+str(c)+" completed in "+str(round(eptime,3))+" seconds.")
    print("The average loss was "+str(round(float(los),3)))#" ("+str(round(float(100*los/baselos),3))+" % of the baseline loss)")
    print(str(round(timeprop*100,1))+"% of the time was spent loading data.")
    print(str(round((N-c)*eptime/60,2))+" minutes remaining.")
    losses += [math.log(los)]
    baseLosses += [baselos]
    epochs += [c]
    if c%A == 0:
        #model.drawVectors()
        Alosses += [sum(losses[-A:])/len(losses[-A:])]
        Aepochs += [c]
        #print(epochs)
        #print(losses)
        axs.plot(epochs,losses,epochs,baseLosses,Aepochs,Alosses)
        fig.canvas.draw()
        fig.canvas.flush_events()
        torch.save(model,save_path+"epoch"+str(epochs[-1]))
        print("model saved.")
    c += 1
axs.plot(epochs,losses,Aepochs,Alosses)
print("done!")

