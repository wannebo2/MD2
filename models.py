#This file will contain the models and all their parts

#TODO:
# - finish fixing coordinate transforms in RoPE pos embeddings
# - Figure out what values to use for the kernal attention approximation, review that to see if I actually implemented it right
# - Update MonarchTransformer to use linear attn. (still need to add random vector drawing procedure into MonarchTransformer)
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
        self.L = initfactor*(torch.rand((b,m,p))-1)  #TODO: fix this - see https://arxiv.org/html/2406.06248v1#S3 
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
    def forward(self,Q,K,V):
        #Q and K should probably be normalized
        
        #Q is of shape (d,h,L)
        #K is of shape (d,h,L)
        #V is of shape (v,L)
        # f2(w,K) maps (d,h,L) to (r,h,L)
        # (r,L)*(L,d) -> (r,d)
        kv = torch.matmul(self.f2(self.W,K),V)
        # then f1(w,Q) maps (d,h,L) to (r,h,L)
        # and (L,h,r)*(r,d) -> (L,h,d)
        qkv = torch.matmul(torch.transpose(self.f1(self.W,Q),0,2),kv)
        #Then, normalize.... the statement below is wrong, I need to figure that bit out.
        #qkvNormalized = torch.div(qkv,torch.sum(qkv,0))
        return qkv#qkvNormalized
    def defaultF1(self,Ws,Qs): #correct if Ws is normalized, which it should be
        #Takes matrix of shape (r,d) and matrix of shape (d,h,L) and returns matrix of shape (r,h,L)
        return self.D*torch.exp(self.A+(self.B*torch.matmul(Ws,Qs))+(self.C*torch.matmul(torch.transpose(Qs,0,2),Qs)))
        # a measurse of simularity between each vector in the list Ws and each key in the list Qs
    def defaultF2(self,Ws,Qs):
        # a measurse of simularity between each vector in the list Ws and each key in the list Ks
        #same as F1 but with additional coefficient self.s
        return self.D*torch.exp(self.A+(self.B*torch.matmul(Ws,Qs))+(self.C*self.s*torch.matmul(torch.transpose(Qs,0,2),Qs)))
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

class RoPE(nn.Module):
    #Positional embedding by rotating the query and key vectors
    def __init__(self,sections,freqs):
        self.sections = sections #sections is a list of splice objects
        self.freqs = freqs
        #In the query vector, sections 0, 1, and 2 are transformed like x, y, and z, respectivley. Section 3 is the rest of the query vector.
    def forward(self,keyEmbeddings,queryEmbeddings,locations):
        #Currently using an inefficient implementation. TODO: figure out or find better algorithim
        for c in range(len(embeddings)): #for each embedding
            key = keyEmbeddings[c]
            #First, perform an actual, 3d, opposite rotations on the query and key vectors so that they align with the absolute coordinate frame
            #This coordinate transformation will probably need some debugging
            query = queryEmbeddings[c]
            qx,qy,qz = query[sections[0]],query[sections[1]],query[sections[2]]
            qx,qy = self.performRotation(self,qx,qy,-1*locations[4]) #First, rotate about the z axis
            qx,qz = self.performRotation(self,qx,qz,-1*locations[5]) #Then, rotate about the y axis
            qy,qz = self.performRotation(self,qy,qz,-1*locations[6]) #Finally rotate about the x axis
            
            queryEmbeddings[c] = torch.cat([qx,qy,qz,query[sections[3]]])
            
            key = keyEmbeddings[c]
            kx,ky,kz = key[sections[0]],key[sections[1]],key[sections[2]]
            ky,kz = self.performRotation(self,ky,kz,locations[6]) #First rotate about the x axis
            kx,kz = self.performRotation(self,kx,kz,locations[5]) #Then rotate about y
            kx,ky = self.performRotation(self,kx,ky,locations[4]) #Finally, rotate about the z axis

            keyEmbeddings[c] = torch.cat([kx,ky,kz,key[sections[3]]])
            #qxs = ((query[sections[0]]*torch.sin(locations[4]))+(query[sections[1]]*torch.cos(locations[4]))*torch.sin(locations[5]))+(query[sections[2]]*torch.cos(locations[5]))
            #qys = ((-1*query[sections[1]]*torch.sin(locations[4]))+(query[sections[0]]*torch.cos(locations[4]))*torch.sin(locations[5]))+(query[sections[2]]*torch.cos(locations[5]))
           # qzs = ((query[sections[1]]*torch.sin(locations[4]))+(query[sections[0]]*torch.cos(locations[4]))*torch.cos(locations[5]))+(-1*query[sections[2]]*torch.sin(locations[5]))
            #query = torch.concat([qxs,qys,qzs,query[sections[3]]])
           
            #Next, perform fake rotations on the keys based on their location information, at frequencies given in self.freqs. Also, perform the opposite rotations on the query vector.
            c2 = 0
            for xi in range(len(locations)):
                for i in range(len(self.freqs)):
                    keyEmbeddings[c2] = (torch.cos(spFreqs[i]*locations[xi])*keyEmbeddings[c2])+(torch.sin(spFreqs[i]*locations[xi])*keyEmbeddings[c2+1])
                    keyEmbeddings[c2+1] = (torch.cos(spFreqs[i]*locations[xi])*keyEmbeddings[c2+1])-(torch.sin(spFreqs[i]*locations[xi])*keyEmbeddings[c2])
                    queryEmbeddings[c2] = (torch.cos(spFreqs[i]*locations[xi])*queryEmbeddings[c2])-(torch.sin(spFreqs[i]*locations[xi])*queryEmbeddings[c2+1])
                    queryEmbeddings[c2+1] = (torch.cos(spFreqs[i]*locations[xi])*queryEmbeddings[c2+1])+(torch.sin(spFreqs[i]*locations[xi])*queryEmbeddings[c2])
                    c2 += 2
        return keyEmbeddings,queryEmbeddings
    def performRotation(self,A,B,angle):
        newA = (torch.cos(angle)*A)+(torch.sin(angle)*B)
        newB = (torch.cos(angle)*B)-(torch.sin(angle)*A)
        return newA,newB
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
        self.acti = nn.LeakyReLu(0.1)
        self.Qnormalizer = torch.nn.LayerNorm((self.heads,self.qkdim))
        self.Knormalizer = torch.nn.LayerNorm((self.qkdim))
        self.Vnormalizer = torch.nn.LayerNorm((self.vdim))
        self.PosModule = RoPE(sections,freqs)
        self.AttnModule = KernalAttention(r,self.qkdim)
    def forward(self,data,locations):
        N = len(data)
        #data should be of the shape (N,m*b), where N is the number of atoms/positions
        # (N,m*b) -> (N,p*q)
        QKV = self.acti(self.QKVlayer(data))
        # (N,p*q) -> [(qkdim*heads,N),(qkdim,N),(vdim,N)]
        QKV = torch.split(torch.t(QKV),[self.qkdim*self.heads,self.qkdim,self.vdim])
        # (qkdim*heads,N) -> (N,heads,qkdim)
        Q = torch.reshape(torch.t(QKV[0]),(N,self.heads,self.qkdim))
        Q = self.Qnormalizer(Q)
        # (qkdim,N) -> (N,qkdim)
        K = torch.t(QKV[1])
        K = self.Knormalizer(K)
        # (vdim,N) -> (N,vdim)
        V = torch.t(QKV[2])
        V = self.Vnormalizer(V)
        # apply relative postion embeddings to Q and K
        K,Q = RoPE(K,Q,locations)
        # compute output of attention layer
        O = self.AttnModule(Q,K,V)
        return O
        

        
