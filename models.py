#This file will contain the models and all their parts

#TODO:
# - finish code enough to run a dummy model
# - get some simulation data to test the dummy model (it shouldn't matter what)
# - try to run a dummy model
# - fix all the errors that prevent the dummy model from running
# - figure out how to correctly initialize the monarch layers (does it matter w/ normalization??)
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
    def __init__(self,m,b,p,q,reshape_output = True,reshape_input = True,initfactor = 0.5): #input is m*b, output is p*q, there are probably some numbers that run best on GPUs, but I do not know what those numbers are.
        # the number of parameters is pb(m+q)
        # in the original paper, m = b = p = q = sqrt(n), so has 2sqrt(n)*n parameters
        # for simular performance with dissimilar input/output sizes, I bet m = b and p = q is the way to go
        # everything should probably be powers of two.
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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
        self.dtype = dtype
        self.L = initfactor*(torch.rand((b,m,p))-1)  #TODO: fix this - see https://arxiv.org/html/2406.06248v1#S3 
        self.R = initfactor*(torch.rand((p,b,q))-1)
        self.reshape_input = reshape_input
        self.reshape_output = reshape_output
    def forward(data):
        if self.reshape_input:
            data = torch.reshape(data,(len(data),-1,self.m,1,self.b))
        data = torch.transpose(data,-3,-1)
        data = torch.matmul(data,self.L)
        data = torch.transpose(data,-3,-1)
        data = torch.matmul(data,self.R)
        if self.reshape_output:
            data = torch.reshape(data,(len(data),-1,self.p*self.q))
        return data

class KernalAttention(nn.Module): #(attempts to) implement the linear attention mechanisim from https://arxiv.org/pdf/2205.15317, https://arxiv.org/pdf/2009.14794, https://arxiv.org/pdf/2302.00787
    #I am still trying to figure out exactly how this is supposed to work
    #Also, the approximation from the paper is length dependent in some unknown way... I dislike that aspect of this architecture, as well as the odd plateau that appears midway through some of the training diagrams before going away.
    #I conceptually prefer Reformer, but as far as I've seen this performs better in practice, and is computationally cheaper
    def __init__(self,r,d,f1=self.defaultF1,f2=self.defaultF2):
        self.f1 = f1
        self.f2 = f2
        self.d = d
        self.r = r
        self.A = 0.1 #1-4A must be greater than 0. TODO: figure out what the optimal value is supposed to be
        self.s = 1
        self.B = pow(self.s*(1-(4*self.A)),0.5)
        self.C = -(self.s+1)/2
        self.D = pow(pow(1-(4*self.A),0.25),self.d)
        self.drawVectors()
    def forward(self,Q,K,V):
        Q = torch.transpose(Q,
        #Q and K should probably be normalized
        
        #Q is of shape (d,h,L)
        #K is of shape (d,h,L)
        #V is of shape (v,L)
        # f2(w,K) maps (d,h,L) to (r,h,L)
        # (r,L)*(L,d) -> (r,d)
        kv = torch.matmul(self.f2(self.W,K),V)
        # then f1(w,Q) maps (d,h,L) to (r,h,L)
        # and (L,h,r)*(r,d) -> (L,h,d)
        qkv = torch.matmul(torch.transpose(self.f1(self.W,Q),-3,-1),kv)
        #Then, normalize.... the statement below is wrong, I need to figure that bit out.
        #qkvNormalized = torch.div(qkv,torch.sum(qkv,0))
        return qkv#qkvNormalized
    def defaultF1(self,Ws,Qs): #correct if Ws is normalized, which it should be
        #Takes matrix of shape (r,d) and matrix of shape (d,h,L) and returns matrix of shape (r,h,L)
        return self.D*torch.exp(self.A+(self.B*torch.matmul(Ws,Qs))+(self.C*torch.matmul(torch.transpose(Qs,-3,-1),Qs)))
        # a measurse of simularity between each vector in the list Ws and each key in the list Qs
    def defaultF2(self,Ws,Qs):
        # a measurse of simularity between each vector in the list Ws and each key in the list Ks
        #same as F1 but with additional coefficient self.s
        return self.D*torch.exp(self.A+(self.B*torch.matmul(Ws,Qs))+(self.C*self.s*torch.matmul(torch.transpose(Qs,-3,-1),Qs)))
    def drawVectors(self):
        # make a list of r orthagonal vectors, each of the shape (d)? Only exactly orthogonal if r<=d
        #Right now gram-shmit method, in the future should probably be replaced with something faster if we are redrawing vectors a lot or using large d
        vectors = torch.normal(mean = torch.ones((r,self.d)),std = torch.ones((r,self.d)))
        for v in range(r):
            for v2 in range(v%self.d):
                vectors[v] -= torch.dot(vectors[v],vectors[v-v2])*vectors[v2]
                vectors[v] /= pow(torch.dot(vectors[v2],vectors[v-v2]),0.5)+0.0001
        self.W = vectors
        #self.W is a matrix of shape (r,d)

class RoPE(nn.Module):# This is somehow still a work in progess.
    #Positional embedding by rotating the query and key vectors
    def __init__(self,sections,freqs):
        self.sections = sections #sections is a list of splice objects
        self.freqs = freqs
        #In the query vector, sections 0, 1, and 2 are transformed like x, y, and z, respectivley. Section 3 is the rest of the query vector.
        #the length of freqs should be equal to half the size of sections 0, 1, and 2, if everything is to transform appropriatley.
    def forward(self,,queryEmbeddings,locations,rotations):
        #Currently working on figuring out how to get what where.

        #Fist, divide the queries and keys into a set of positional components that rotate, and a stationary part
        rotQ = queryEmbeddings[...,:3,:]
        statQ = queryEmbeddings[...,3:,:]

        rotK = keyEmbeddings[...,:3,:]
        statK = keyEmbeddings[...,:3,:]
        #Rotate the positional parts of the query and key vectors to the global coordinate frame
        # (b,N,3,3) * (b,N,3,d) -> (b,N,3,d)
        rotQ = torch.matmul(rotations,rotQ)
        rotK = torch.matmul(rotations,rotK)
        
        queryEmbeddings = torch.flatten(torch.cat([rotQ,statQ],-2),start_dim=-2)
        keyEmbeddings =  torch.flatten(torch.cat([rotK,statK],-2),start_dim=-2)

        #Create a tensor of shape (b,N,freqs*locations)
        frequencyMultiplied = torch.flatten(torch.einsum("i,ljk->ljik", self.freqs, locations),start_dim=-2) # like below, if this doesn't flatten in the order I need it to, stuff will break
        
        #reshape the embeddings into pairs
        queryEmbeddings = torch.reshape(queryEmbeddings,(len(queryEmbeddings),len(queryEmbeddings[0]),-1,2)) #will need to test to make sure this reshapes data the way I expect it to. It probably will not.
        #make a set of rotation matricies of the shape (2,2,b,N,freqs*locations)
        s = torch.sin(frequencyMultiplied)
        c = torch.cos(frequencyMultiplied)
        fakeRotMat = torch.tensor([[c,s],[-s,c]])
        #apply those rotations to the embeddings
        queryEmbeddings = torch.flatten(torch.einsum("robnd,bndo->bndr",fakeRotMat,queryEmbeddings),start_dim=-2) #it would be funny to make these spell something
        keyEmbeddings = torch.flatten(torch.einsum("robnd,bndo->bndr",fakeRotMat,keyEmbeddings),start_dim=-2)
        return keyEmbeddings,queryEmbeddings

class MonarchTransformer(nn.Module): 
    def __init__(self,m,b,p,q,heads,qkdim,vdim,sections,freqs,r): #input is in shape of (m*b)
        super().__init__()
        self.m = m
        self.b = b
        self.p = p
        self.q = q
        self.heads = heads
        self.qkdim = qkdim
        self.sqqk = pow(qkdim,0.5)
        self.vdim = vdim
        self.posdim = posdim
        self.QKVlayer = MonarchLayer(self.m,self.b,self.p,self.q,reshape_output = True,reshape_input = True)
        self.acti = nn.LeakyReLu(0.1) #not sure this is really the right choice, a smoother one is probably going to be more stable.
        self.Qnormalizer = torch.nn.LayerNorm((self.heads,self.qkdim))
        self.Knormalizer = torch.nn.LayerNorm((self.qkdim))
        self.Vnormalizer = torch.nn.LayerNorm((self.vdim))
        self.PosModule = RoPE(sections,freqs)
        self.AttnModule = KernalAttention(r,self.qkdim)
    def forward(self,data,locations,rotations):
        N = len(data)
        #data should be of the shape (N,m*b), where N is the number of atoms/positions
        # (N,m*b) -> (N,p*q)
        QKV = self.acti(self.QKVlayer(data))
        # (N,p*q) -> [(qkdim*heads,N),(qkdim,N),(vdim,N)]
        QKV = torch.split(torch.t(QKV,-2,-1),[self.qkdim*self.heads,self.qkdim,self.vdim])
        # (qkdim*heads,N) -> (N,heads,qkdim)
        Q = torch.reshape(torch.t(QKV[:,0],-2,-1),(len(data),-1,self.heads,self.qkdim))
        Q = self.Qnormalizer(Q)
        # (qkdim,N) -> (N,qkdim)
        K = torch.t(QKV[:,1],-2,-1)
        K = self.Knormalizer(K)
        # (vdim,N) -> (N,vdim)
        V = torch.t(QKV[:,2],-2,-1)
        V = self.Vnormalizer(V)
        # apply relative postion embeddings to Q and K
        K,Q = RoPE(K,Q,locations,rotations)
        # compute output of attention layer
        O = self.AttnModule(Q,K,V)
        return O
    def drawVectors(self):
        self.AttnModule.drawVectors()
        
class outputLayer(nn.Module): # we will need something to convert the model output to a vector of the right shape, and then rotate the output to align with the global coordinate system so that a simple loss function can be used
    #this will be that something
    
    def __init__(self,m,b,p,q):
        self.layer = MonarchLayer(m,b,p,q,reshape_output = True,reshape_input = True,initfactor = 0.5)
    def forward(self,data,locs,rotations):
        data = self.layer(data)
        data = torch.reshape(data,(len(data),len(data[0]),3,-1))
        data = torch.matmul(rotations,data)
        NewLocs = torch.flatten(data[...,:-3],start_dim=-2)
        NewRots = data[...,-3:]
        return NewLocs,NewRots
class DebuggingModel(nn.Module):
    def __init__(self,layers):
        m = 10
        b = 10
        p = 20
        q = 50
        heads = 10
        qkdim = 100
        vdim = 100
        outp = 4
        outq = 3
        sections = [slice(0,10),slice(10,20),slice(20,30),slice(30,-1)]
        freqs = [0.014,0.153,1.877,13.471,116.7] #these are just some random numbers of different orders of magnitude. In the future, these should be chosen more intelligently.
        r = 63
        self.Layers = []
        while len(self.Layers)<layers:
            self.Layers += [MonarchTransformer(self,m,b,p,q,heads,qkdim,vdim,sections,freqs,r)]
        self.OutputLayer = outputLayer(m,b,outp,outq)
    def forward(self,data,locs,rots):
        for l in self.Layers[:-1]:
            data = l(data,locs,rots)
        return self.OutputLayer(data)
        
        
        
