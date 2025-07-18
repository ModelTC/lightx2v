# Standard Library
import hashlib

# Import from third library
import json

# Import from alt
from alt.utils.petrel_helper import global_petrel_helper
from loguru import logger


class BaseModelFactoryInference(object):
    CACHE_PKL = "ceph-sh1424-private:s3://data.infra.sql.cache/model_factory_inference/{}.pkl"

    def __init__(self, use_cache=True) -> None:
        self.use_cache = use_cache

    def generate_unique_token(self, data):
        json_str = json.dumps(data, sort_keys=True)
        json_str = self.__class__.__name__ + "-" + json_str
        hash_obj = hashlib.sha256(json_str.encode())
        token = hash_obj.hexdigest()
        return token

    def process(self, input_paths, **kwargs):
        cache_file = self.CACHE_PKL.format(self.generate_unique_token(input_paths))

        if self.use_cache and global_petrel_helper.check_exists(cache_file):
            logger.info("Use Cache: {}".format(cache_file))
            return global_petrel_helper.load_pk(cache_file)

        res = self.forward(input_paths, **kwargs)

        global_petrel_helper.save_pk(cache_file, res)
        return res

    def forward(self):
        raise NotImplementedError
