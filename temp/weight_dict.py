import torch

ckpt_path = "/mtc/yongyang/models/x2v_models/hunyuan/lightx2v_format/t2v/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
output_path = "temp/weight_keys.txt"

# 加载并提取 module 权重字典
weight_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)["module"]

# 写入所有键名到 txt 文件
with open(output_path, "w") as f:
    for key in weight_dict.keys():
        f.write(key + "\n")

print(f"共写入 {len(weight_dict)} 个参数键到 {output_path}")
