# 步数蒸馏

步数蒸馏是 LightX2V 中的一项重要优化技术，通过训练蒸馏模型将推理步数从原始的 40-50 步大幅减少到 **4 步**，在保持视频质量的同时显著提升推理速度。LightX2V 在实现步数蒸馏的同时也加入了 CFG 蒸馏，进一步提升推理速度。

## 🔍 技术原理

步数蒸馏通过 [Self-Forcing](https://github.com/guandeh17/Self-Forcing) 技术实现。Self-Forcing 针对 1.3B 的自回归模型进行步数蒸馏、CFG蒸馏。LightX2V 在其基础上，进行了一系列扩展：

1. **更大的模型**：支持 14B 模型的步数蒸馏训练；
2. **更多的模型**：支持标准的双向模型，以及 I2V 模型的步数蒸馏训练；

具体实现可参考 [Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus)。

## 🎯 技术特性

- **推理加速**：推理步数从 40-50 步减少到 4 步且无需 CFG，速度提升约 **20-24x**
- **质量保持**：通过蒸馏技术保持原有的视频生成质量
- **兼容性强**：支持 T2V 和 I2V 任务
- **使用灵活**：支持加载完整步数蒸馏模型，或者在原生模型的基础上加载步数蒸馏LoRA

## 🛠️ 配置文件说明

### 基础配置文件

在 [configs/distill/](https://github.com/ModelTC/lightx2v/tree/main/configs/distill) 目录下提供了多种配置选项：

| 配置文件 | 用途 | 模型地址 |
|----------|------|------------|
| [wan_t2v_distill_4step_cfg.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_t2v_distill_4step_cfg.json) | 加载 T2V 4步蒸馏完整模型 | TODO |
| [wan_i2v_distill_4step_cfg.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_i2v_distill_4step_cfg.json) | 加载 I2V 4步蒸馏完整模型 | TODO |
| [wan_t2v_distill_4step_cfg_lora.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_t2v_distill_4step_cfg_lora.json) | 加载 Wan-T2V 模型和步数蒸馏 LoRA | TODO |
| [wan_i2v_distill_4step_cfg_lora.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_i2v_distill_4step_cfg_lora.json) | 加载 Wan-I2V 模型和步数蒸馏 LoRA | TODO |

### 关键配置参数

```json
{
  "infer_steps": 4,                              // 推理步数
  "denoising_step_list": [999, 750, 500, 250],   // 去噪时间步列表
  "enable_cfg": false,                           // 关闭CFG以提升速度
  "lora_path": [                                 // LoRA权重路径（可选）
    "path/to/distill_lora.safetensors"
  ]
}
```

## 📜 使用方法

### 模型准备

**完整模型：**
将下载好的模型（`distill_model.pt` 或者 `distill_model.safetensors`）放到 Wan 模型根目录的 `distill_models/` 文件夹下即可
- 对于 T2V：`Wan2.1-T2V-14B/distill_models/`
- 对于 I2V-480P：`Wan2.1-I2V-14B-480P/distill_models/`

**LoRA：**
1. 将下载好的 LoRA 放到任意位置
2. 修改配置文件中的 `lora_path` 参数为 LoRA 存放路径即可

### 推理脚本

**T2V 完整模型：**
```bash
bash scripts/wan/run_wan_t2v_distill_4step_cfg.sh
```

**I2V 完整模型：**
```bash
bash scripts/wan/run_wan_i2v_distill_4step_cfg.sh
```

### 步数蒸馏 LoRA 推理脚本

**T2V LoRA：**
```bash
bash scripts/wan/run_wan_t2v_distill_4step_cfg_lora.sh
```

**I2V LoRA：**
```bash
bash scripts/wan/run_wan_i2v_distill_4step_cfg_lora.sh
```

## 🔧 服务化部署

### 启动蒸馏模型服务

对 [scripts/server/start_server.sh](https://github.com/ModelTC/lightx2v/blob/main/scripts/server/start_server.sh) 中的启动命令进行修改：

```bash
python -m lightx2v.api_server \
  --model_cls wan2.1_distill \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/distill/wan_t2v_distill_4step_cfg.json \
  --port 8000 \
  --nproc_per_node 1
```

运行服务启动脚本：

```bash
scripts/server/start_server.sh
```

更多详细信息见[服务化部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_service.html)。

### 在 Gradio 界面中使用

见 [Gradio 文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_gradio.html)
