import torch

w1 = torch.load("/mtc/yongyang/models/x2v_models/hunyuan/lightx2v_format/i2v/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt")["module"]
w2 = torch.load("/mtc/gushiqiao/llmc_workspace/x2v_models/hunyuan/hunyuan_i2v_int8.pth")["module"]

print(len(w1.keys()))
print(len(w2.keys()))
for key in w1.keys():
    print(key)
    print(w1[key].dtype, w2[key].dtype)
