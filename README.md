# LightX2V: Light Video Generation Inference Framework 

<div align="center">
  <picture>
    <img alt="LightLLM" src="assets/img_lightx2v.jpg" width=75%>
  </picture>
</div>

--------------------------------------------------------------------------------


## START ENV

```
docker pull registry.cn-sh-01.sensecore.cn/devsft-ccr-2/video-gen:25033101

docker run --gpus all -itd --ipc=host --name [name] -v /mnt:/mnt --entrypoint /bin/bash [image id]
```

## START RUN

```
git clone https://gitlab.bj.sensetime.com/video-gen/lightx2v.git
cd lightx2v/scripts

# Modify the parameters of the running script
bash run_hunyuan_t2v.sh
```
