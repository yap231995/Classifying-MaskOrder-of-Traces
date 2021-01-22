import scipy.io
import os
mat0 = scipy.io.loadmat('traces_mask_order0.mat')
mat1 = scipy.io.loadmat('traces_mask_order1.mat')

os.system("python3 gen_mask_traces.py")

print(mat0["traces"].shape)
print(mat0["labels"])

print(mat1["traces"].shape)
print(mat1["labels"])