#This file will contain the models and all their parts

#TODO:
# - Figure out what functions to use in the kernal attention approximation
# - Implement chosen linear-ish transformer
# - figure out how to use 8-bit Galore
# - write training (and testing) pipeline
# - add visualization
# - train and debug!
# - if we need more data, figure out how to talk to NAMD/write program to automatically generate relavant simulations

import torch
from torch import nn
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
        self.L = initfactor*(torch.rand((b,m,p))-1)
        self.R = initfactor*(torch.rand((p,b,q))-1)
        self.reshape_input = reshape_input
        self.reshape_output = reshape_output
    def forward(data):
        if self.reshape_input:
            data = torch.reshape(data,(self.m,1,self.b))
        data = torch.transpose(data,0,2)
        data = torch.bmm(data,self.L)
        data = torch.transpose(data,0,2)
        data = torch.bmm(data,self.R)
        if self.reshape_output:
            data = torch.reshape(data,(self.p*self.q))
        return data

class KernalAttention(nn.Module): #(attempts to) implement the linear attention mechanisim from https://arxiv.org/pdf/2205.15317, https://arxiv.org/pdf/2009.14794, https://arxiv.org/pdf/2302.00787
    #I am still trying to figure out exactly how this is supposed to work
    #Also, the approximation from the paper is length dependent in some unknown way... I dislike that aspect of this architecture, as well as the odd plateau that appears midway through some of the training diagrams before going away.
    #I conceptually prefer Reformer, but as far as I've seen this performs better in practice, and is computationally cheaper
    def __init__(self,r,f1=self.defaultF1,f2=self.defaultF2):
        self.f1 = f1
        self.f2 = f2
        self.r = r
    def forward(self,Q,K,V):
        # f2(w,K) maps (d,L) to (r,L)
        # (r,L)*(L,d) -> (r,d)
        kv = torch.matmul(self.f2(self.W,K),V)
        # then f1(w,Q) maps (d,h,L) to (r,h,L)
        # and (L,h,r)*(r,d) -> (L,h,d)
        qkv = torch.matmul(torch.transpose(self.f1(self.W,Q),0,2),kv)
        #Then, normalize along L axis so that entires sum to one (needed b/c variable length input)
        qkvNormalized = torch.div(qkv,torch.sum(qkv,0))
        return qkvNormalized
    def defaultF1(self,Ws,Qs):
        # a measurse of simularity between each vector in the list Ws and each key in the list Qs
    def defaultF2(self,Ws,Qs):
        # a measurse of simularity between each vector in the list Ws and each key in the list Qs
    def drawVectors(self,d,L):
        # make a list of r orthagonal vectors, each of the shape (d)?
        # wait that can't be right
class MonarchTransformer(nn.Module): #at the moment, I am going to design it to support concatenating position vectors as opposed to adding them, because that's how I feel like it should work
    # position embeddings will be delt with somewhere else.
    # designed to be used with monarch layers without reshaping.
    # If training is unstable, it is probably because of something wrong in here
    def __init__(self,m,b,heads,qkdim,vdim,posdim): #input is in shape of (m,b)
        super().__init__()
        self.m = m
        self.b = b
        self.p = heads
        self.heads = heads
        self.qkdim = qkdim
        self.sqqk = pow(qkdim,0.5)
        self.vdim = vdim
        self.posdim = posdim
        self.Qlayer = MonarchLayer(self.m,self.b,self.qkdim+self.posdim,self.heads,reshape_output = False,reshape_input = False)
        self.KVlayer = MonarchLayer(self.m,self.b,1,self.qkdim+self.vdim,reshape_output = False,reshape_input = False)
        self.acti = nn.LeakyReLu(0.1)
        self.attentionActi = nn.Softmax(dim=1) #I am not sure the dimension is right... will have to double-check that.
    def forward(self,data,Ks,Vs):
        #Ks is of shape (N,qkdim+posdim)
        #Vs is of shape (N,vdim)
        
        #data = nn.functional.normalize(data) #nvrmnd, putting normalization in model instead of layer
        
        #Ks is a matrix of the shape (N,qkdim), Vs is a matrix of the shape (N,vdim)
        # (m,1,b) -> (qkdim+posdim,heads)
        Q = self.acti(self.Qlayer(data))
        Q = torch.reshape(Q,(self.qkdim+self.posdim,self.heads))
        # (m,1,b) -> (qkdim) and (vdim)
        KV = self.acti(self.KVlayer(data))
        KV = torch.reshape(KV,(self.qkdim+self.vdim))
        KV = torch.split(KV,[self.qkdim,self.vdim])
        # (N,qkdim+posdim)x(qkdim+posdim,heads) ->(N,heads)
        QK = self.attentionActi(torch.mm(Ks,Q))
        # (N,heads) -> (heads,N)
        KQ = torch.transpose(QK,0,1)
        # (heads,N)x(N,vdim) -> (heads,vdim)
        Result = torch.mm(KQ,Vs)
        #Returns (result of Q), (K),(V)
        # (heads,vdim), (qkdim), (vdim)
        return Result,KV[0],KV[1]
        
class Equivariant_Module(nn.Module):
    #Contains a monarch transformer and a feed-forward network, and performs the appropriate coordinate transformations on the positional embeddings.
    #need to figure out exactly what the transformation is.
    def __init__(self,inp_embed_m,inp_embed_b,oup_embed_p,oup_embed_q,kqdim,posdim):
        self.inp_embed_m = inp_embed_m
        self.inp_embed_b = inp_embed_b
        self.oup_embed_p = inp_embed_p
        self.oup_embed_q = oup_embed_q
        self.kqdim = kqdim
        self.posdim = posdim
        self.Transformer = MonarchTransformer(inp_embed_m,inp_embed_b,oup_embed_p,kqdim,oup_embed_q,posdim)
        self.FF = MonarchLayer(inp_embed_m,inp_embed_b,oup_embed_p,oup_embed_q,reshape_output = False,reshape_input = False)
    def forward(locs,tree):
        
