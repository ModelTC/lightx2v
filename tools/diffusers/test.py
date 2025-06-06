from safetensors import safe_open
import torch


w1 = torch.load("/mnt/afs_2/gushiqiao/x2v_new_models/t5_int8.pth")
w2 = torch.load("/mnt/afs_2/gushiqiao/x2v_new_models/t5_int8/t5_int8.pth")

# print(w1['blocks.20.attn.k.weight_scale'])
# print(w2['blocks.20.attn.k.weight_scale'])

for k in w1.keys():
    print((w1[k] - w2[k]).float().abs().max())
