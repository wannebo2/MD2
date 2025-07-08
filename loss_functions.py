import torch
def compute_loss(outputLocs,outputRots,expectedLocs,expectedRots,noisedLocs,noisedRots):#,disp): #putting this in it's own file feels a little silly now that I've done it.
    #di = noisedLocs[...,:-1]-expectedLocs[...,:-1]
    #edisp = torch.pow(torch.einsum("bnc,bnc -> bn",di,di),0.5)
    LocScore = torch.mean(torch.pow(outputLocs[...,:-1]-expectedLocs[...,:-1],2)/(torch.pow(noisedLocs[...,:-1]-expectedLocs[...,:-1],2)+0.00001))
    RotScore = torch.mean(torch.pow(outputRots-expectedRots,2)/(torch.pow(noisedRots-expectedRots,2)+0.00001))
    #DispScore = torch.mean(torch.pow(edisp-disp,2)/(torch.pow(edisp,2)+0.001))
    return (1*LocScore)#+(1*RotScore)
