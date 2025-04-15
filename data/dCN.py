import numpy as np

def CN(dist):
    return 1/(1 + np.exp(-16*(1.889726*2/dist - 1)))

def distance(x1,y1,z1,x2,y2,z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def dCN(x1,y1,z1,x2,y2,z2):
    dist = distance(x1,y1,z1,x2,y2,z2)
    return CN(dist)**2 * -16 * np.exp(-16*(1.889726*2/dist - 1)) * (1.889726*2/dist**3)

x1 = 5.137/0.529
y1 = 5.551/0.529
z1 = 10.1047/0.529
x2 = 4.5168/0.529
y2 = 6.1365/0.529
z2 = 11.36043/0.529

diff = 0.001/0.529

cn1 = CN(distance(x1,y1,z1,x2,y2,z2))
cn2 = CN(distance(x1+diff,y1,z1,x2,y2,z2))
dcn = dCN(x1,y1,z1,x2,y2,z2)

print("CN:", cn1)
print("distance:", distance(x1,y1,z1,x2,y2,z2))
print("CN2:", cn2)
print("distance2:", distance(x1+diff,y1,z1,x2,y2,z2))
print("dCN:", dcn*(x1-x2))
print("diff in CN:", (cn2 - cn1)/diff)