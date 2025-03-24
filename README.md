# lightx2v

这是一个视频生成推理框架


## 运行环境

```
# 内网镜像
docker pull registry.cn-sh-01.sensecore.cn/devsft-ccr-2/video-gen:25031303

docker run --gpus all -itd --ipc=host --name [name] -v /mnt:/mnt --entrypoint /bin/bash [image id]
```

## 运行方式

```
git clone https://gitlab.bj.sensetime.com/video-gen/lightx2v.git
cd lightx2v

# 修改运行脚本的参数
bash run_hunyuan_t2v.sh
```
