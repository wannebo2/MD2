import torch
def compute_loss(outputLocs,outputRots,expectedLocs,expectedRots): #putting this in it's own file feels a little silly now that I've done it.
    return torch.sum(torch.pow(outputLocs-expectedLocs,2))+torch.sum(torch.pow(outputRots-expectedRots,2))
