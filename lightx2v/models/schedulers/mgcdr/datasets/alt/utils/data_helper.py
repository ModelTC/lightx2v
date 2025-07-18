import json
from decimal import Decimal
import numpy as np


class AltJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, Decimal):
            return float(obj)

        if isinstance(obj, np.int64):
            return int(obj)

        if isinstance(obj, set):
            return list(obj)

        return json.JSONEncoder.default(self, obj)


class AltDataWrapper(object):
    @classmethod
    def pkl_to_json(self, pkl_meta):
        _json = json.loads(json.dumps(pkl_meta, cls=AltJsonEncoder))
        return _json
