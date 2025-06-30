import torch
import models
import dataloading
import loss_functions
def train(data_loader,model):
    for i,data in enumerate(data_loader):
        stoof,expLocs,expRots = data
        atmEmbeds = stoof[0]
        noisedLocs = stoof[1]
        noisedRots = stoof[2]
        print("a")
        optimizer.zero_grad()
        outLocs,outRots = model(atmEmbeds,noisedLocs,noisedRots)
        print("b")
        loss = loss_functions.compute_loss(outLocs,outRots,expLocs,expRots)
        print("c")
        loss.backward()
        print("cycle")
        if i % 100 == 0:
            print(str(round(i/l,3)*100)+" % complete")
            print(loss)
    print("done!")
#so far heavily based on the tutorial https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
path = "trajectories/"
dataset = dataloading.TrajectoryDataset(path)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=3,shuffle=True)
model = models.DebuggingModel(3)
#model.cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
l = len(data_loader)
train(data_loader,model)

