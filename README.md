# LightX2V: Light Video Generation Inference Framework

<div align="center">
  <picture>
    <img alt="LightLLM" src="assets/img_lightx2v.jpg" width=75%>
  </picture>
</div>

--------------------------------------------------------------------------------

## Supported Model List

✅ [HunyuanVideo-T2V](https://huggingface.co/tencent/HunyuanVideo)

✅ [HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V)

✅ [Wan2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

✅ [Wan2.1-I2V](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)

✅ [Wan2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)

## Fast Start Up With Conda

```shell
# clone repo and submodules
git clone https://github.com/ModelTC/lightx2v.git lightx2v && cd lightx2v
git submodule update --init --recursive

# create conda env and install requirments
conda create -n lightx2v python=3.11 && conda activate lightx2v
pip install -r requirements.txt

# install flash-attention 2
RUN cd lightx2v/3rd/flash-attention && pip install --no-cache-dir -v -e .

# install flash-attention 3, only if hopper
RUN cd lightx2v/3rd/flash-attention/hopper && pip install --no-cache-dir -v -e .

# modify the parameters of the running script
bash scripts/run_hunyuan_t2v.sh
```

## Fast Start Up With Docker

```shell
docker pull lightx2v/lightx2v:latest
docker run -it --rm --name lightx2v --gpus all --ipc=host lightx2v/lightx2v:latest
```

## Contributing Guidelines

We have prepared a `pre-commit` hook to enforce consistent code formatting across the project. If your code complies with the standards, you should not see any errors, you can clean up your code following the steps below:

1. Install the required dependencies:

```shell
pip install ruff pre-commit
```

2. Then, run the following command before commit:

```shell
pre-commit run --all-files
```

3. Finally, please double-check your code to ensure it complies with the following additional specifications as much as possible:
  - Avoid hard-coding local paths: Make sure your submissions do not include hard-coded local paths, as these paths are specific to individual development environments and can cause compatibility issues. Use relative paths or configuration files instead.
  - Clear error handling: Implement clear error-handling mechanisms in your code so that error messages can accurately indicate the location of the problem, possible causes, and suggested solutions, facilitating quick debugging.
  - Detailed comments and documentation: Add comments to complex code sections and provide comprehensive documentation to explain the functionality of the code, input-output requirements, and potential error scenarios.

Thank you for your contributions!
