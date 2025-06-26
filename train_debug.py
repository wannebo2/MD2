import torch
import models
import dataloader

#so far heavily based on the tutorial https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
path = "trajectories/"
dataset = dataloader.TrajectoryDataset(path)
data_loader = torch.utils.DataLoader(dataset,batch_size=3,shuffle=True)
model = models.DebuggingModel(3)


def train():
    for i,data in enumerate(training_loader):
        stoof,expectedLocs = data
        atmEmbeds = stoof[0]
        noisedLocs = stoof[1]
        optimizer.zero_grad()
        outputs = model(atmEmbeds,noisedLocs)
        
        loss = loss_fn(outputs,
