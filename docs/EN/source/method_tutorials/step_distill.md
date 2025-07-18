# Step Distillation

Step distillation is an important optimization technique in LightX2V. By training distilled models, it significantly reduces inference steps from the original 40-50 steps to **4 steps**, dramatically improving inference speed while maintaining video quality. LightX2V implements step distillation along with CFG distillation to further enhance inference speed.

## 🔍 Technical Principle

Step distillation is implemented through [Self-Forcing](https://github.com/guandeh17/Self-Forcing) technology. Self-Forcing performs step distillation and CFG distillation on 1.3B autoregressive models. LightX2V extends it with a series of enhancements:

1. **Larger Models**: Supports step distillation training for 14B models;
2. **More Model Types**: Supports standard bidirectional models and I2V model step distillation training;

For detailed implementation, refer to [Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus).

## 🎯 Technical Features

- **Inference Acceleration**: Reduces inference steps from 40-50 to 4 steps without CFG, achieving approximately **20-24x** speedup
- **Quality Preservation**: Maintains original video generation quality through distillation techniques
- **Strong Compatibility**: Supports both T2V and I2V tasks
- **Flexible Usage**: Supports loading complete step distillation models or loading step distillation LoRA on top of native models

## 🛠️ Configuration Files

### Basic Configuration Files

Multiple configuration options are provided in the [configs/distill/](https://github.com/ModelTC/lightx2v/tree/main/configs/distill) directory:

| Configuration File | Purpose | Model Address |
|-------------------|---------|---------------|
| [wan_t2v_distill_4step_cfg.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_t2v_distill_4step_cfg.json) | Load T2V 4-step distillation complete model | TODO |
| [wan_i2v_distill_4step_cfg.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_i2v_distill_4step_cfg.json) | Load I2V 4-step distillation complete model | TODO |
| [wan_t2v_distill_4step_cfg_lora.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_t2v_distill_4step_cfg_lora.json) | Load Wan-T2V model and step distillation LoRA | TODO |
| [wan_i2v_distill_4step_cfg_lora.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_i2v_distill_4step_cfg_lora.json) | Load Wan-I2V model and step distillation LoRA | TODO |

### Key Configuration Parameters

```json
{
  "infer_steps": 4,                              // Inference steps
  "denoising_step_list": [999, 750, 500, 250],   // Denoising timestep list
  "enable_cfg": false,                           // Disable CFG for speed improvement
  "lora_path": [                                 // LoRA weights path (optional)
    "path/to/distill_lora.safetensors"
  ]
}
```

## 📜 Usage

### Model Preparation

**Complete Model:**
Place the downloaded model (`distill_model.pt` or `distill_model.safetensors`) in the `distill_models/` folder under the Wan model root directory:
- For T2V: `Wan2.1-T2V-14B/distill_models/`
- For I2V-480P: `Wan2.1-I2V-14B-480P/distill_models/`

**LoRA:**
1. Place the downloaded LoRA in any location
2. Modify the `lora_path` parameter in the configuration file to the LoRA storage path

### Inference Scripts

**T2V Complete Model:**
```bash
bash scripts/wan/run_wan_t2v_distill_4step_cfg.sh
```

**I2V Complete Model:**
```bash
bash scripts/wan/run_wan_i2v_distill_4step_cfg.sh
```

### Step Distillation LoRA Inference Scripts

**T2V LoRA:**
```bash
bash scripts/wan/run_wan_t2v_distill_4step_cfg_lora.sh
```

**I2V LoRA:**
```bash
bash scripts/wan/run_wan_i2v_distill_4step_cfg_lora.sh
```

## 🔧 Service Deployment

### Start Distillation Model Service

Modify the startup command in [scripts/server/start_server.sh](https://github.com/ModelTC/lightx2v/blob/main/scripts/server/start_server.sh):

```bash
python -m lightx2v.api_server \
  --model_cls wan2.1_distill \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/distill/wan_t2v_distill_4step_cfg.json \
  --port 8000 \
  --nproc_per_node 1
```

Run the service startup script:

```bash
scripts/server/start_server.sh
```

For more details, see [Service Deployment](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_service.html).

### Usage in Gradio Interface

See [Gradio Documentation](https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_gradio.html)
