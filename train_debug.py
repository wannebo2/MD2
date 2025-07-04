import torch
import models
import dataloading
import loss_functions
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random
def train(data_loader,model,c):
    global device
    losses = []
    I = 0
    Kl = random.randint(0,199)
    Kl2 = random.randint(0,6)
    Kr2 = random.randint(0,2)
    Kr3 = random.randint(0,2)
    for i,data in enumerate(data_loader):
        stoof,expLocs,expRots = data
        atmEmbeds = stoof[0]
        noisedLocs = stoof[1]
        noisedRots = stoof[2]
        atmEmbeds = atmEmbeds.to(device)
        noisedLocs = noisedLocs.to(device)
        noisedRots = noisedRots.to(device)
        expLocs = expLocs.to(device)
        expRots = expRots.to(device)
        #print("a")
        outLocs,outRots = model(atmEmbeds,noisedLocs,noisedRots)
        print("location rnd value: "+str(outLocs[0,Kl,Kl2]))
        print("rotation rnd value: "+str(outRots[0,Kr2,Kr3]))
        #print("b")
        #loss = loss_functions.compute_loss(outLocs,outRots,expLocs,expRots)
        #losses.append(torch.sum(loss).cpu().detach()/(5*200*(7+9)))
        #print("c")
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()
       # print("cycle")
        I += 1
        #if i % 100 == 0:
            #print(str(round(i/l,3)*100)+" % complete with epoch number "+str(c))
            #print(loss)
    return 0#sum(losses)/I
    print("done!")
path = "trajectories/"
print("loading dataset...")
dataset = dataloading.TrajectoryDataset(path)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
print("initializing model...")
model = models.DebuggingModel(3)
if torch.cuda.is_available:
    device = torch.device('cuda')
    print("runing on CUDA")
else:
    device = torch.device('cpu')
    print("running on CPU")
model.to(device)
print("initializing optimizer...")
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
l = len(data_loader)
N = 1000
print("training for "+str(N)+" epochs")
c = 0
plt.ion()
fig, axs = plt.subplots(1, 1)
axs.set_xlabel("epoch")
axs.set_ylabel("average loss per output")
axs.grid(True)
axs.yaxis.set_minor_locator(MultipleLocator(10))
axs.xaxis.set_minor_locator(MultipleLocator(10))
Alosses = []
losses = []
epochs = []
Aepochs = []
while c<N:
    losses += [train(data_loader,model,c)]
    epochs += [c]
    c += 1
    if c%100 == 0:
        #model.drawVectors()
        Alosses += [sum(losses[-100:])/len(losses[-100:])]
        Aepochs += [c]
        axs.plot(epochs,losses,Aepochs,Alosses)
        fig.canvas.draw()
        fig.canvas.flush_events()
axs.plot(epochs,losses,Aepochs,Alosses)
print("done!")

