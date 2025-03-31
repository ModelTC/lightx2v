# LightX2V: Light Video Generation Inference Framework

<div align="center">
  <picture>
    <img alt="LightLLM" src="assets/img_lightx2v.jpg" width=75%>
  </picture>
</div>

--------------------------------------------------------------------------------

## Prepare Environment

```shell
docker pull registry.cn-sh-01.sensecore.cn/devsft-ccr-2/video-gen:25033101
docker run --gpus all -itd --ipc=host --name [name] -v /mnt:/mnt --entrypoint /bin/bash [image id]
```

## Fast Start

```shell
git clone https://github.com/ModelTC/lightx2v.git
cd lightx2v

# Modify the parameters of the running script
bash run_hunyuan_t2v.sh
```

## Contribute

We have prepared a `pre-commit` hook to enforce consistent code formatting across the project. You can clean up your code following the steps below:

1. Install the required dependencies:

```shell
    pip install ruff pre-commit
```

2. Then, run the following command:

```shell
    pre-commit run --all-files
```

If your code complies with the standards, you should not see any errors.
