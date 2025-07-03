#This file will contain the models and all their parts

#TODO:
# - Figure out what values to use for the kernal attention approximation, review that to see if I actually implemented it right
# - add redrawing procedure to the kernal attn approx.
# - figure out if the normalization layers are actually in the right places
# - figure out how to use 8-bit Galore, or if it is not applicable 
# - write training (and testing) pipeline
# - don't forget to actually set the atom embeddings, because it currently uses random ones
# - add visualization
# - train and debug!
# - if we need more data, figure out how to talk to NAMD/write program to automatically generate relavant simulations

import torch
from torch import nn
import general_utils
#for how layers work, see the pytorch tutorial https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
#global datatype should be set if desired. 
class MonarchLayer(nn.Module):
    # structured version of a dense layer
    # See https://arxiv.org/abs/2204.00595 for why
    def __init__(self,m,b,p,q,reshape_output = True,reshape_input = True,initfactor = 1): #input is m*b, output is p*q, there are probably some numbers that run best on GPUs, but I do not know what those numbers are.
        # the number of parameters is pb(m+q)
        # in the original paper, m = b = p = q = sqrt(n), so has 2sqrt(n)*n parameters
        # for simular performance with dissimilar input/output sizes, I bet m = b and p = q is the way to go
        # everything should probably be powers of two.
        super().__init__()
        #self.input_dim = input_dim
        #self.output_dim = output_dim
        # reshape : (mb) -> (m,1,b)
        # P1 : (m,1,b) -> (b,1,m)
        # L : (b,1,m)x(b,m,p) -> (b,1,p)
        # P2 : (b,1,p) -> (p,1,b)
        # R : (p,1,b)x(p,b,q) -> (p,1,q)
        # reshape : (p,1,q) -> (pq)
        self.m = m
        self.b = b
        self.p = p
        self.q = q
        #self.dtype = dtype
        self.L = nn.Parameter(initfactor*(torch.normal(mean = torch.zeros(b,m,p))/m))  #TODO: fix this initialization - see https://arxiv.org/html/2406.06248v1#S3 
        self.R = nn.Parameter(initfactor*(torch.normal(mean = torch.zeros(p,b,q))/b)) # I think i've fixed them, but the gradients may need to be rescaled or something
        self.Biases = nn.Parameter(initfactor*(torch.normal(mean = torch.zeros(p,1,q))))
        self.reshape_input = reshape_input
        self.reshape_output = reshape_output
    def forward(self,data):
        if self.reshape_input:
            data = torch.reshape(data,(len(data),-1,self.m,1,self.b))
        data = torch.transpose(data,-3,-1)
        #print("data device: "+str(data.get_device()))
        #print("L device: "+str(self.L.get_device()))
       # print("R device: "+str(self.R.get_device()))
        data = torch.matmul(data,self.L)
        data = torch.transpose(data,-3,-1)
        data = torch.matmul(data,self.R)
        data = data+self.Biases
        if self.reshape_output:
            data = torch.reshape(data,(len(data),-1,self.p*self.q))
        return data

class KernalAttention(nn.Module): #(attempts to) implement the linear attention mechanisim from https://arxiv.org/pdf/2205.15317, https://arxiv.org/pdf/2009.14794, https://arxiv.org/pdf/2302.00787
    #I am still trying to figure out exactly how this is supposed to work
    #Also, the approximation from the paper is length dependent in some unknown way... I dislike that aspect of this architecture, as well as the odd plateau that appears midway through some of the training diagrams before going away.
    #I conceptually prefer Reformer, but as far as I've seen this performs better in practice, and is computationally cheaper
    def __init__(self,r,d,f1=None,f2=None):
        super().__init__()
        if f1 == None:
            self.f1 = self.defaultF1#f1
        else:
            self.f1 = f1
        if f2 == None:
            self.f2 = self.defaultF1 #f2
        else:
            self.f2 = f2
        self.d = d
        self.r = r
        self.A = 0.05#nn.Parameter(torch.tensor(0.05)) #1-4A must be greater than 0. TODO: figure out what the optimal value is supposed to be
        self.s = 1
        self.B = pow(self.s*(1-(4*self.A)),0.5)
        self.C = -(self.s+1)/2
        self.D = pow(pow(1-(4*self.A),0.25),self.d)
        self.device = 0
        self.drawVectors()
    def to(self,device):
        self.device = device
        #self.A = self.A.to(device)
        self.W = self.W.to(device)
        #self.drawVectors()
    def forward(self,Q,K,V):
        #Q = torch.transpose(Q,
        #Q and K should probably be normalized
        #print("Q shape: "+str(Q.shape))
       # #print("K shape: "+str(K.shape))
        #print("V shape: "+str(V.shape))
        #print("W shape: "+str(self.W.shape))
        #Q is of shape (b,d,h,L)
        #K is of shape (b,d,h,L)
        #V is of shape (b,v,L)
        # f2(w,K) maps (d,h,L) to (r,h,L)
        # (r,L)*(L,d) -> (r,d)
        wk = self.f2(self.W,K)
        #print("wk shape: "+str(wk.shape))
        kv = torch.einsum("bnr,bnd->brd",wk,V)
        #print("KV shape: "+str(kv.shape))
        # then f1(w,Q) maps (d,h,L) to (r,h,L)
        # and (L,h,r)*(r,d) -> (L,h,d)
        wq = self.f1(self.W,Q)
        #print("WQ shape: "+str(wq.shape))
        qkv = torch.einsum("bnhr,brd->bnhd",wq,kv)
        #qkv = torch.matmul(torch.transpose(wq,-3,-1),kv)
        #Then, normalize.... the statement below is wrong, I need to figure that bit out.
        #qkvNormalized = torch.div(qkv,torch.sum(qkv,0))
        return qkv#qkvNormalized
    def defaultF1(self,Ws,Qs): #correct if Ws is normalized, which it should be
        #Takes matrix of shape (r,d) and matrix of shape (d,h,L) and returns matrix of shape (r,h,L)
        wk = torch.matmul(Qs,Ws)
        return torch.exp((wk))#torch.matmul(torch.transpose(Qs,-3,-1),Qs)))
        # a measurse of simularity between each vector in the list Ws and each key in the list Qs
    def defaultF1(self,Ws,Qs): #correct if Ws is normalized, which it should be
        # same as F1 but with an extra factor self.s
        wk = torch.matmul(Qs,Ws)
        return torch.exp((wk))#torch.matmul(torch.transpose(Qs,-3,-1),Qs)))
        
    def drawVectors(self):
        # make a list of r orthagonal vectors, each of the shape (d)? Only exactly orthogonal if r<=d
        #Right now gram-shmit method, in the future should probably be replaced with something faster if we are redrawing vectors a lot or using large d
        vectors = torch.normal(mean = torch.zeros((self.r,self.d)),std = torch.ones((self.r,self.d)))
        for v in range(self.r):
            for v2 in range(v%self.d):
                vectors[v] -= torch.dot(vectors[v],vectors[v-v2])*vectors[v2]
                vectors[v] /= pow(torch.dot(vectors[v],vectors[v]),0.5)+0.0001
        self.W = nn.Parameter(torch.t(vectors))
        #self.W is a matrix of shape (r,d)

class RoPE(nn.Module):
    #Positional embedding by rotating the query and key vectors
    def __init__(self,freqs,sections):
        super().__init__()
        self.freqs = torch.tensor(freqs)
        self.U = sections
    def to(self, device):
        super().to(device)
        #print("I'm being called!")
        self.freqs = self.freqs.to(device)
    def forward(self,keyEmbeddings,queryEmbeddings,locations,rotations):
        
        #First, reshape the key and queries into a more convienient shape
        queryEmbeddings = torch.reshape(queryEmbeddings,(len(queryEmbeddings),len(queryEmbeddings[0]),len(queryEmbeddings[0][0]),self.U,-1))
        keyEmbeddings = torch.reshape(keyEmbeddings,(len(keyEmbeddings),len(keyEmbeddings[0]),self.U,-1))
        #Fist, divide the queries and keys into a set of positional components that rotate, and a stationary part
        rotQ = queryEmbeddings[...,:3,:]
        statQ = queryEmbeddings[...,3:,:]

        rotK = keyEmbeddings[...,:3,:]
        statK = keyEmbeddings[...,3:,:]
        #Rotate the positional parts of the query and key vectors to the global coordinate frame
        # (b,N,3,3) * (b,N,h,3,d) -> (b,N,h,3,d)
        rotQ = torch.einsum("bnac,bnhcd->bnhad",rotations,rotQ)
        rotK = torch.matmul(rotations,rotK)
        
        queryEmbeddings = torch.flatten(torch.cat([rotQ,statQ],-2),start_dim=-2)
        keyEmbeddings =  torch.flatten(torch.cat([rotK,statK],-2),start_dim=-2)

        #Create a tensor of shape (b,N,freqs*locations)
        #print("f device: "+str(self.freqs.device))
        #print("loc device: "+str(locations.device))
        frequencyMultiplied = torch.flatten(torch.einsum("i,ljk->ljik", self.freqs, locations),start_dim=-2) # like below, if this doesn't flatten in the order I need it to, stuff will break
        
        #reshape the embeddings into pairs
        queryEmbeddings = torch.reshape(queryEmbeddings,(len(queryEmbeddings),len(queryEmbeddings[0]),len(queryEmbeddings[0][0]),-1,2)) #will need to test to make sure this reshapes data the way I expect it to. It probably will not.
        keyEmbeddings = torch.reshape(keyEmbeddings,(len(keyEmbeddings),len(keyEmbeddings[0]),-1,2))
        #make a set of rotation matricies of the shape (2,2,b,N,freqs*locations)
        s = torch.sin(frequencyMultiplied)
        c = torch.cos(frequencyMultiplied)
        fakeRotMat = torch.stack([torch.stack([c,s]),torch.stack([-1*s,c])])
        #apply those rotations to the embeddings
        #print("rot mat shape: "+str(fakeRotMat.shape))
        #print("Q shape: "+str(queryEmbeddings.shape))
        #print("K shape: "+str(keyEmbeddings.shape))
        rotQ = queryEmbeddings[...,:fakeRotMat.shape[-1],:]
        statQ = queryEmbeddings[...,fakeRotMat.shape[-1]:,:]
        rotK = keyEmbeddings[...,:fakeRotMat.shape[-1],:]
        #print("rotK shape: "+str(rotK.shape))
        statK = keyEmbeddings[...,fakeRotMat.shape[-1]:,:]
        rotQ = torch.einsum("robnd,bnhdo->bnhdr",fakeRotMat,rotQ) #it would be funny to make these spell something
        rotK = torch.einsum("robnd,bndo->bndr",fakeRotMat,rotK)
        queryEmbeddings = torch.flatten(torch.cat([rotQ,statQ],-2),start_dim=-2)
        keyEmbeddings = torch.flatten(torch.cat([rotK,statK],-2),start_dim=-2)
        return keyEmbeddings,queryEmbeddings

class MonarchTransformer(nn.Module): 
    def __init__(self,m,b,p,q,heads,qkdim,vdim,freqs,r,sections): #input is in shape of (m*b)
        super().__init__()
        self.m = m
        self.b = b
        self.p = p
        self.q = q
        self.heads = heads
        self.qkdim = qkdim
        self.sqqk = pow(qkdim,0.5)
        self.vdim = vdim
        #self.posdim = posdim
        self.QKVlayer = MonarchLayer(self.m,self.b,self.p,self.q,reshape_output = True,reshape_input = True)
        self.acti = nn.SELU()
        #self.Qnormalizer = torch.nn.LayerNorm((self.heads,self.qkdim))
        #self.Knormalizer = torch.nn.LayerNorm((self.qkdim))
        #self.Vnormalizer = torch.nn.LayerNorm((self.vdim))
        self.PosModule = RoPE(freqs,sections)
        self.AttnModule = KernalAttention(r,self.qkdim)
        self.FFlayer = MonarchLayer(self.m,self.b,self.heads,self.vdim,reshape_output = True,reshape_input = True)
        self.OutputNorm = torch.nn.LayerNorm((self.heads*self.vdim))
    def forward(self,data,locations,rotations):
        N = len(data)
        #data should be of the shape (N,m*b), where N is the number of atoms/positions
        # (N,m*b) -> (N,p*q)
        QKV = self.acti(self.QKVlayer(data))
        # (N,p*q) -> [(qkdim*heads,N),(qkdim,N),(vdim,N)]
        QKV = torch.split(QKV,[self.qkdim*self.heads,self.qkdim,self.vdim],dim=-1)
        # (qkdim*heads,N) -> (N,heads,qkdim)
        Q = torch.reshape(QKV[0],(len(data),-1,self.heads,self.qkdim))
        #print("Q shape: "+str(Q.shape))
        divsr = torch.unsqueeze(torch.pow(torch.sum(torch.pow(Q,2),dim=-1),0.5),dim=-1)+0.0001
       # print("divisor shape: "+str(divsr.shape))
        Q = torch.div(Q,divsr)
        # (qkdim,N) -> (N,qkdim)
        K = QKV[1]#torch.transpose(QKV[1],-2,-1)
       # print("K shape: "+str(K.shape))
        divsr = torch.unsqueeze(torch.pow(torch.sum(torch.pow(K,2),dim=-1),0.5),dim=-1)+0.0001
        K = torch.div(K,divsr)
        # (vdim,N) -> (N,vdim)
        V = QKV[2]#torch.transpose(QKV[2],-2,-1)
       # print("V shape: "+str(K.shape))
        divsr = torch.unsqueeze(torch.pow(torch.sum(torch.pow(V,2),dim=-1),0.5),dim=-1)+0.0001
        V = torch.div(V,divsr)
        # apply relative postion embeddings to Q and K
        K,Q = self.PosModule(K,Q,locations,rotations)
        # compute output of attention layer
        O = self.AttnModule(Q,K,V)
        O = torch.flatten(O,start_dim = -2)
        #print("max O: "+str(torch.max(O)))
       # print("O shape: "+str(O.shape))
        #and the output of the feed forward layer
        FFO = self.acti(self.FFlayer(data))
       # print("max FFO: "+str(torch.max(FFO)))
       # print("FFO shape: "+str(FFO.shape))
        out = self.OutputNorm(O+FFO)
       # print("max out: "+str(torch.max(out)))
        return out#return their sum
    def drawVectors(self):
        self.AttnModule.drawVectors()
    def to(self,device):
        super().to(device)
        #print("I've been called.")
        self.FFlayer.to(device)
        self.QKVlayer.to(device)
        self.AttnModule.to(device)
        self.PosModule.to(device)
class outputLayer(nn.Module): # we will need something to convert the model output to a vector of the right shape, and then rotate the output to align with the global coordinate system so that a simple loss function can be used
    #this will be that something
    
    def __init__(self,m,b,p,q):
        super().__init__()
        self.mLayer = MonarchLayer(m,b,p,q,reshape_output = True,reshape_input = True,initfactor = 0.5)
    def forward(self,data,locs,rotations):
        data = self.mLayer(data)
        data = torch.reshape(data,(len(data),len(data[0]),4,-1))
        rotData = torch.matmul(rotations,data[...,:-1,:])
       # print("rotData shape: "+str(rotData.shape))
        NewLocs = torch.flatten(rotData[...,:2],start_dim=-2)
        NewRots = rotData[...,2:5]+rotations
        NewLocs = torch.cat([NewLocs,locs[...,-1:]],dim=-1)+locs
       # NewRots = data[...,-9:]
    
      #  print("NewLocs shape: "+str(NewLocs.shape))
       # print("NewRots shape: "+str(NewRots.shape))
        return NewLocs,NewRots
class DebuggingModel(nn.Module):
    def __init__(self,layers):
        super().__init__()
        m = 100
        b = 10
        p = 20
        q = 60
        u = 10
        heads = 10
        qkdim = 100
        vdim = 100
        outp = 4
        outq = 5
        sections = [slice(0,10),slice(10,20),slice(20,30),slice(30,-1)]
        freqs = [0.014,0.153,1.877,13.471,116.7] #these are just some random numbers of different orders of magnitude. In the future, these should be chosen more intelligently.
        r = 63
        self.Layers = []
        while len(self.Layers)<layers:
            self.Layers += [MonarchTransformer(m,b,p,q,heads,qkdim,vdim,freqs,r,u)]
        self.Layers = nn.ModuleList(self.Layers)
        self.OutputLayer = outputLayer(m,b,outp,outq)
    def forward(self,data,locs,rots):
        for i,l in enumerate(self.Layers):
            #print("running layer...")
            #print(" data shape is "+str(data.shape))
            data = l(data,locs,rots)
        return self.OutputLayer(data,locs,rots)
    def drawVectors(self):
        for i,l in enumerate(self.Layers):
            l.drawVectors()
    def to(self,device):
        super().to(device)
        for i,l in enumerate(self.Layers):
            l.to(device)
        self.OutputLayer.to(device)
        
        
        
