# LightX2V: Light Video Generation Inference Framework

<div align="center" id="lightx2v">
<img alt="logo" src="assets/img_lightx2v.png" width=75%></img>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ModelTC/lightx2v)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://lightx2v-en.readthedocs.io/en/latest)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://hub.docker.com/r/lightx2v/lightx2v/tags)

</div>

--------------------------------------------------------------------------------

## Supported Model List

✅ [HunyuanVideo-T2V](https://huggingface.co/tencent/HunyuanVideo)

✅ [HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V)

✅ [Wan2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

✅ [Wan2.1-I2V](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)

✅ [Wan2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)

✅ [SkyReels-V2-DF](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P)

✅ [CogVideoX1.5-5B-T2V](https://huggingface.co/THUDM/CogVideoX1.5-5B)

## How to Run

Please refer to the [documentation](https://github.com/ModelTC/lightx2v/tree/main/docs) in lightx2v.

## Contributing Guidelines

We have prepared a `pre-commit` hook to enforce consistent code formatting across the project.

1. Install the required dependencies:

```shell
pip install ruff pre-commit
```

2. Then, run the following command before commit:

```shell
pre-commit run --all-files
```

Thank you for your contributions!


## Acknowledgments

We built the code for this repository by referencing the code repositories involved in all the models mentioned above.
