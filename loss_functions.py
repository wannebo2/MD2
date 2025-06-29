def compute_loss(ouputLocs,outputRots,expectedLocs,expectedRots): #putting this in it's own file feels a little silly now that I've done it.
    return torch.sum(torch.power(outputLocs-expectedLocs,2)+torch.power(outputRots-expectedRots,2))
