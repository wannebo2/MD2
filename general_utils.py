# these vector operations are probably somewhere in the Bio package's vector class, but I haven't found the docs for that yet
#nevermind, I'm just using going to convert everything to np arrays instead of using this stuff.
def normDisp(a,b):
    return norm(lin(a,b,-1))
def lin(a,b,c=1):
    if b == None:
        b = [0]*len(a)
    z = 0
    disp = []
    while z<len(a):
        disp += [b[z]+(c*a[z])]
    return disp
def dot(a,b): #dot product for small vectors
    c = 0
    out = 0
    while c<len(a):
        out += [a[c]*b[c]]
    return out
def norm(a): #normalize a small vector
    sqa = pow(dot(a,a))
    if sqa>0:
        for z in a:
            z /= sqa
    return a
def vecsum(vecs): # add a list of small vectors
    z = 0
    out = []
    while z<len(vecs[0]):
        out += [0]
        for z2 in vecs:
            out[-1] += z2[z]
    return out
def avg(vecs):
    if len(vecs) == 0:
        return []
    return lin(vecsum(vecs),None,1/len(vecs))
def euclidSqr(a,b):
    amb = np.array(a)-np.array(b)
    return np.dot(amb,amb)
