# Standard Library
import os
import pickle as pkl
import subprocess

# Import from third library
import json
from tqdm import tqdm

# Import from alt
import yaml

# Import from local
from .bee_helper import BeeUploader
from .petrel_helper import PetrelHelper, expand_s3_path, global_petrel_helper
from .yaml_helper import IncludeLoader

aws_ak = "O5B34FE51MMZRZGEY1Z5"
aws_sk = "Xh6hOX5xnZ6BrsNnCcCt8NzNCbWeCSYwtABErKJ6"


def _prepare_folder(path):
    if path.count("/") > 0:
        folder = path.rsplit("/", 1)[0]
        os.makedirs(folder, exist_ok=True)
    else:
        pass


def load(path):
    if "s3://" not in path:
        assert os.path.exists(path), path
    if path.endswith(".yaml"):
        out = yaml.load(open(path, "r"), Loader=IncludeLoader)
    elif path.endswith(".pkl"):
        out = global_petrel_helper.load_pk(path, mode="rb")
    elif path.endswith(".json"):
        out = global_petrel_helper.load_json(path)
    elif path.endswith(".csv"):
        out = global_petrel_helper.read_csv(path)
    else:
        raise NotImplementedError
    return out


def dump(path, obj):
    if "s3://" not in path:
        _prepare_folder(path)
    if path.endswith(".pkl"):
        pkl.dump(obj, open(path, "wb"))
    elif path.endswith(".json"):
        global_petrel_helper.save_json(path, obj)
    elif path.endswith(".yaml"):
        yaml.dump(obj, open(path, "w"))
    else:
        raise NotImplementedError


def load_autolabel_meta_json(path):
    path = expand_s3_path(path)
    json_meta = global_petrel_helper.load_json(path)
    if "calib_geometry" in json_meta["data_structure"]["config"]:
        json_meta["data_structure"]["config"] = json_meta["data_structure"]["config"].replace("calib_geometry", "calib")

    json_meta = global_petrel_helper.map_bucket(json_meta)
    assert json_meta is not None
    return json_meta


def dump_json_lines(output, lst):
    with PetrelHelper.open(output, "w") as writer:
        total_num = len(lst)

        if isinstance(lst, (tuple, list)):
            lst = enumerate(lst)
        elif isinstance(lst, dict):
            lst = lst.items()

        for _, js in tqdm(lst, total=total_num, desc=f"dumping to {output}"):
            writer.write(json.dumps(js, ensure_ascii=False) + "\n")


def ads_cli_download_folder(src, dst, ads_cli_listers=5, ads_cli_threads=5, aws_ak=None, aws_sk=None):
    bucket, folder = src.strip().split("//")[-1].split("/", 1)

    if aws_ak is None or aws_sk is None:
        aws_config = BeeUploader.get_aksk(BeeUploader.get_cluster(f"s3://{bucket}"))
        aws_ak, aws_sk = aws_config["accessID"], aws_config["accessSecret"]

    cmd = f"ads-cli -l {ads_cli_listers} -p {ads_cli_threads} sync s3://{aws_ak}:{aws_sk}@{bucket}.auto-business.st-sh-01.sensecoreapi-oss.cn/{folder}/ {dst}/"
    subprocess.run(cmd, shell=True, check=True)


def ads_cli_upload_folder(
    src,
    dst,
    ads_cli_listers=5,
    ads_cli_threads=5,
    aws_ak=aws_ak,
    aws_sk=aws_sk,
    endpoint="http://auto-business.st-sh-01.sensecoreapi-oss.cn",
):
    bucket, folder = dst.strip().split("//")[-1].split("/", 1)

    endpoint = endpoint.strip("http://")
    # cmd = f'ads-cli -l {ads_cli_listers} -p {ads_cli_threads} sync {src}/ s3://{aws_ak}:{aws_sk}@{bucket}.sdc-oss.iagproxy.senseauto.com/{folder}/'
    cmd = f"ads-cli -l {ads_cli_listers} -p {ads_cli_threads} sync {src}/ s3://{aws_ak}:{aws_sk}@{bucket}.{endpoint}/{folder}/"
    subprocess.run(cmd, shell=True, check=True)
