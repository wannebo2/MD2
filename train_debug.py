import torch
import models
import dataloader
import loss_functions
#so far heavily based on the tutorial https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
path = "trajectories/"
dataset = dataloader.TrajectoryDataset(path)
data_loader = torch.utils.DataLoader(dataset,batch_size=3,shuffle=True)
model = models.DebuggingModel(3)
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

def train():
    opti
    for i,data in enumerate(training_loader):
        stoof,expectedLocs = data
        atmEmbeds = stoof[0]
        noisedLocs = stoof[1]
        
        optimizer.zero_grad()
        outputs = model(atmEmbeds,noisedLocs)
        noisedLocs = 
        loss = loss_functions.compute_loss(outputs,noisedLocs,expectedLocs)
        loss.backward()
