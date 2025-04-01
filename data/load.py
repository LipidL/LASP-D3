import numpy as np
import os
import struct
import torch

d3_params = np.load(f"{os.path.abspath(__file__)}/../dftd3_params.npz")
print(f"keys: {list(d3_params.keys())}")
print(f"shape of c6ab: {d3_params['c6ab'].shape}") # (95,95,5,5,3)
# meaning: element pair between 95 elements, 5 different CN values each.
# 1st value: C_6^{AB} reference value of the pair of elements A and B. it's less or equal than 0 if no value is available.
# 2nd value: coordination number of 1st atom (A) in the pair.
# 3rd value: coordination number of 2nd atom (B) in the pair. 
print(f"shape of r0ab: {d3_params['r0ab'].shape}")
print(f"shape of rcov: {d3_params['rcov'].shape}")
print(f"shape of r2r4: {d3_params['r2r4'].shape}") # used to calculate c8 from c6
# Save d3_params to a binary file for direct mapping in C/C++
with open(f"{os.path.abspath(__file__)}/../params.bin", "wb") as f:
    for key in ["c6ab", "r0ab", "rcov", "r2r4"]:
        data = d3_params[key]
        data = data.astype(np.float32).copy(order='C')  # Convert to C-contiguous array of bytes
        # Write the shape of the array as metadata
        f.write(struct.pack("I", len(data.shape)))  # Number of dimensions (32-bit unsigned integer)
        f.write(struct.pack("I" * len(data.shape), *data.shape))  # Shape
        # Write the array data
        f.write(data.tobytes())

c6ab = torch.tensor(d3_params["c6ab"])
data = d3_params["c6ab"]
print(f"c6ab between Po: {data[84, 84]}")

