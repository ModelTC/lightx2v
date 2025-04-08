# LightX2V: Light Video Generation Inference Framework

<div align="center">
  <picture>
    <img alt="LightLLM" src="assets/img_lightx2v.jpg" width=75%>
  </picture>
</div>

--------------------------------------------------------------------------------

## Fast Start Up

```shell
git clone https://github.com/ModelTC/lightx2v.git lightx2v && cd lightx2v

conda create -n lightx2v python=3.11 && conda activate lightx2v
pip install -r requirements.txt

# download flash attention and install
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention && pip install -v -e .
# for FA3, cd flash-attention/hopper && pip install -v -e .

# modify the parameters of the running script
bash scripts/run_hunyuan_t2v.sh
```

## Docker Image

```shell
docker pull lightx2v/lightx2v:latest
docker run -it --rm --gpus all --ipc=host lightx2v/lightx2v:latest
```

## Contribute

We have prepared a `pre-commit` hook to enforce consistent code formatting across the project. You can clean up your code following the steps below:

1. Install the required dependencies:

```shell
    pip install ruff pre-commit
```

2. Then, run the following command before commit:

```shell
    pre-commit run --all-files
```

If your code complies with the standards, you should not see any errors.
