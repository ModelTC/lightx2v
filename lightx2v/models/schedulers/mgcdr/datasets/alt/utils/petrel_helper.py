# Standard Library
import configparser
import functools
import io
import os
import pickle as pk
import re
import boto3

# Import from third library
import cv2
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import from alt
import addict
# from alt.utils.pcd_helper import point_cloud_from_buffer
from loguru import logger


def petreloss_path():
    if os.getenv("PETRELPATH", ""):
        return os.getenv("PETRELPATH", "")
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    if os.getenv("ALT_NET_INSIDE", "off").upper() == "ON":
        return os.path.join(father_path, "/mnt/iag/user/tangweixuan/aoss.conf")
    else:
        return os.path.join(father_path, "/mnt/iag/user/tangweixuan/aoss.conf")


def expand_s3_path(path):
    if not path.startswith('s3://'):
        return path
    elif path.startswith('s3://sdc_gt_label'):
        return 'ad_system_common:' + path
    elif path.startswith('s3://sdc3_gt_label'):
        return 'ad_system_common_oss3:' + path
    elif path.startswith('s3://sdc3-gt-label-2'):
        return 'ad_system_common_auto:' + path
    else:
        raise NotImplementedError(path)


class PetrelOpen(object):
    def __init__(self, filename, mode="r", **kwargs):
        self.mode = mode
        if "r" in mode:
            self.handle = PetrelHelper._petrel_helper.load_data(filename, **kwargs)
        elif "w" in mode:
            self.handle = PetrelWriter(filename, PetrelHelper._petrel_helper.client)
        else:
            print(mode)

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc_value, exc_trackback):
        if self.mode == "w":
            self.handle.close()
        del self.mode
        del self.handle


class PetrelWriter(object):
    def __init__(self, filename, client=None):
        self.filename = filename
        self.client = client if client else PetrelHelper._petrel_helper.client

        self.buf = io.BytesIO()

    def write(self, info):
        self.buf.write(info.encode("utf-8"))

    def close(self):
        self.client.put(self.filename, self.buf.getvalue())


class PetrelHelper(object):
    _petrel_helper = None
    open = PetrelOpen
    BUCKET_PROG = re.compile(r"^s3://{(?P<bucket>(.*?))}/(?P<key>(.*?))$")
    ALTER_HOST = "alter_base"
    S3URI_TMPLTS = addict.Dict(origin=r"s3://{bucket}/{key}", profile=r"{cluster}:{s3uri}")

    def __init__(self, conf_path=petreloss_path()):
        self.conf_path = conf_path
        self.conf = self.parse_conf_file(conf_path)

        self._inited = False
        self._init_petrel()
        PetrelHelper._petrel_helper = self

    @staticmethod
    def parse_conf_file(conf_path):
        """Parse .ini file"""
        parser = configparser.ConfigParser(default_section=None)
        parser.read(conf_path)
        cfgdict = dict(map(lambda section: (section[0], addict.Dict(**section[1])), parser.items()))
        cfgdict.pop(None)
        return cfgdict

    def _init_petrel(self):
        try:
            # Import from alt
            # from petrel_client.client import Client
            from aoss_client.client import Client

            self.client = Client(self.conf_path)

            self._inited = True
        except Exception as e:
            print("init petrel failed: {}, ".format(e))

    def bytes_to_img(self, value):
        img_array = np.frombuffer(value, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert img is not None
        return img

    def save_checkpoint(self, model, path):
        if "s3://" not in path:
            torch.save(model, path)
        else:
            with io.BytesIO() as f:
                torch.save(model, f)
                f.seek(0)
                self.client.put(path, f)

    def load_pretrain(self, path, map_location=None):
        if "s3://" not in path:
            assert os.path.exists(path), f"No such file: {path}"
            return torch.load(path, map_location=map_location)
        elif "http://" in path:
            return torch.hub.load_state_dict_from_url(path, map_location=map_location)
        else:
            self.check_init()

            file_bytes = self.client.get(path)
            buffer = io.BytesIO(file_bytes)
            res = torch.load(buffer, map_location=map_location)
            return res

    def imread(self, filename):
        value = self.client.Get(filename)
        assert value is not None, filename
        img = self.bytes_to_img(value)
        return img

    def imwrite(self, filename, image, ext=".jpg"):
        if "s3://" not in filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            cv2.imwrite(filename, image)
            return
        assert filename.endswith(ext)
        success, array = cv2.imencode(ext, image)
        assert success
        img_bytes = array.tostring()
        try:
            self.client.put(filename, img_bytes)
        except Exception as e:  # noqa
            print(e)
            print("save to {} failed".format(filename))

    def read_img(self, filename):
        return self.imread(filename)

    def check_init(self):
        if not self._inited:
            raise Exception("petrel oss not inited")

    def readlines(self, path):
        if "s3://" in path:
            if not self._inited:
                self._init_petrel()
            for line in self._iter_ceph_lines(path):
                yield line
        else:
            for line in open(path).readlines():
                yield line

    def _iter_ceph_lines(self, path):
        response = self.client.get(path, enable_stream=True, no_cache=True)

        for line in response.iter_lines():
            cur_line = line.decode("utf-8")
            yield cur_line

    def load_data(self, path, ceph_read=True, fs_read=False, mode="r"):
        if "s3://" not in path:
            if not fs_read:
                return open(path, mode)
            else:
                return open(path, mode).read()
        else:
            self.check_init()

            if ceph_read:
                return self._iter_cpeh_lines(path)
            else:
                return self.client.get(path)

    def save_json(self, path, data):
        if "s3://" not in path:
            json.dump(data, open(path, "w"), indent=4, ensure_ascii=False)
        else:
            dump_data = json.dumps(data).encode("utf-8")
            self.client.put(path, dump_data)

    def save_jsonl(self, path, data):
        if "s3://" not in path:
            with open(path, "w") as f:
                for item in tqdm(data, total=len(data), desc=f"dumping to {path}"):
                    f.writelines(json.dumps(item) + "\n")
        else:
            with PetrelHelper.open(path, "w") as writer:
                total_num = len(data)
                for _, js in tqdm(enumerate(data), total=total_num, desc=f"dumping to {path}"):
                    writer.write(json.dumps(js, ensure_ascii=False) + "\n")

    def save_pk(self, path, data):
        if "s3://" not in path:
            pk.dump(data, open(path, "wb"))
        else:
            self.client.put(path, pk.dumps(data))

    @staticmethod
    def load_pk(path, mode="r"):
        if "s3://" not in path:
            pk_res = pk.load(open(path, mode))
        else:
            pk_res = pk.loads(PetrelHelper._petrel_helper.load_data(path, ceph_read=False))
        return pk_res

    @staticmethod
    def load_json(path, mode="r"):
        if "s3://" not in path:
            js = json.load(open(path, mode))
        else:
            js = json.loads(PetrelHelper._petrel_helper.load_data(path, ceph_read=False))
        return js

    def listdir(self, path):
        if "s3://" not in path:
            return os.listdir(path)
        else:
            return self.client.list(path)

    def isdir(self, path):
        if "s3://" not in path:
            return os.path.isdir(path)
        else:
            return self.client.isdir(path)

    def check_exists(self, url):
        exists = self.client.contains(url)
        return exists

    def read(self, path):
        ext = path.split(".")[-1]
        return getattr(self, "read_{}".format(ext))(path)

    def map_bucket(self, meta, bucket_mapper=None, skip_check=False):
        """copy from auto-labeling/autolabel/apis/load.py"""
        if meta is None:
            return meta
        if bucket_mapper is None:
            bucket_mapper = meta.pop("bucket_mapper", None)
        if bucket_mapper is None:
            return meta
        map_func = functools.partial(self.map_bucket, bucket_mapper=bucket_mapper)
        if isinstance(meta, str):
            match = self.BUCKET_PROG.match(meta)
            if match is None:
                return meta.rstrip("/")
            bkt_aka = match.group("bucket")
            mapper = bucket_mapper[bkt_aka].copy()
            bucket = mapper.pop("name")
            endpoints = set(mapper.values())
            s3uri = meta.format(**{bkt_aka: bucket}).rstrip("/")
            cluster_match = None
            for cluster, kwargs in self.conf.items():
                if kwargs.get("host_base", None) is None:
                    continue
                if kwargs.host_base in endpoints:
                    cluster_match = cluster
                    break
            if cluster_match is None:
                raise TypeError(f"invalid path: {meta}, cannot find the profile of {bkt_aka}")
            path = self.S3URI_TMPLTS.profile.format(cluster=cluster_match, s3uri=s3uri)
            if not skip_check and not (self.client.isdir(path) or self.client.contains(path)):
                logger.warning(f"path does not exist: {path}")
                return None
            return path.rstrip("/")
        if isinstance(meta, (list, tuple)):
            return list(map(map_func, meta))
        if isinstance(meta, dict):
            return dict(
                map(lambda _meta: (_meta[0], map_func(_meta[1], skip_check=_meta[0] == "data_annotation")), meta.items())
            )
        raise TypeError(f"invalid meta: {meta}")

    def load_bin(self, url):
        bin_bytes = self.client.get(url, no_cache=True)
        if bin_bytes is None:
            return None
        bin_array = np.frombuffer(bin_bytes, dtype=np.float32).reshape(-1, 4)
        return bin_array

    def read_csv(self, path, **kwargs):
        if "s3://" not in path:
            data_frame = pd.read_csv(path, **kwargs)
        else:
            bytes = self.client.get(path, no_cache=True)
            data_frame = pd.read_csv(io.BytesIO(bytes), **kwargs)
        return data_frame

    def load_pcd(self, url):
        value = self.client.Get(url)
        if value == b"":
            return False, None
        assert value is not None, url
        points = point_cloud_from_buffer(value)
        if "intensity" not in points.pc_data.dtype.names:
            pcd_ceph = [points.pc_data["x"], points.pc_data["y"], points.pc_data["z"], np.ones((points.pc_data["x"].shape[0]))]
        else:
            pcd_ceph = [points.pc_data["x"], points.pc_data["y"], points.pc_data["z"], points.pc_data["intensity"]]
        pc_velo = np.asarray(pcd_ceph).T
        return True, pc_velo

    def save_bin(self, url, data):
        self.client.put(url, data.tobytes())

    def load_text(self, url):
        if "s3://" in url and not self.client.contains(url):
            return ""
        else:
            file_bytes = self.client.get(url)
            return file_bytes.decode("utf-8")

    def get_bucket_key(self, s):
        if s.startswith('s3://'):
            prefix = self.conf['DEFAULT']['default_cluster']
        else:
            prefix = s.split(':s3:')[0]

        lst = s.split('/')
        if lst[0][-1] == ':':
            bucket = lst[2]
            key = '/'.join(lst[3:])
        else:
            bucket = ''
            key = '/'.join(lst)
        return prefix, bucket, key

    def get_url(self, video_path, timeout=864000000):
        prefix, bucket, key = self.get_bucket_key(video_path)

        client = boto3.client(service_name='s3',
                              aws_access_key_id=self.conf[prefix]['access_key'],
                              aws_secret_access_key=self.conf[prefix]['secret_key'],
                              endpoint_url=self.conf[prefix]['host_base'],
                              verify=False)

        url = client.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=timeout)

        return url


global_petrel_helper = PetrelHelper()


if __name__ == "__main__":
    video_path = "ad_system_common:s3://sdc_gt_label/ql_test/sensebee/pvb/pvb_all_attr/0522/18_meta_2024_03_21_06_42_00_gacGtParser/visualize/22205_2024_03_21_06_42_00_gacGtParser.mp4"
    video_url = global_petrel_helper.get_url(video_path=video_path)
    print(video_url)
