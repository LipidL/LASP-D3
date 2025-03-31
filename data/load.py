import numpy as np
import torch

d3_params = np.load("dftd3_params.npz")
print(f"keys: {list(d3_params.keys())}")
print(f"shape of c6ab: {d3_params['c6ab'].shape}") # (95,95,5,5,3)
# meaning: element pair between 95 elements, 5 different CN values each.
# 1st value: C_6^{AB} reference value of the pair of elements A and B. it's less or equal than 0 if no value is available.
# 2nd value: coordination number of 1st atom (A) in the pair.
# 3rd value: coordination number of 2nd atom (B) in the pair. 
print(f"shape of r0ab: {d3_params['r0ab'].shape}")
print(f"shape of rcov: {d3_params['rcov'].shape}")
print(f"shape of r2r4: {d3_params['r2r4'].shape}") # used to calculate c8 from c6

print(f"parameters for Cl: {d3_params['c6ab'][17][17]}")

c6ab = torch.tensor(d3_params["c6ab"], dtype=torch.float32)
cn0, cn1, cn2 = c6ab.reshape(-1, 5, 5, 3).split(1, dim=3)
print(f"cn0: {cn0.shape}, cn1: {cn1.shape}, cn2: {cn2.shape}")
