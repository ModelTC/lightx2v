import json
import os
from alt.utils import load, AltDataWrapper, DictWrapper

if os.getenv("ALT_RICH_PRING", "ON").upper() == "ON":
    from rich import print


class AltPklModule(object):
    def __init__(self, path) -> None:
        self.pkl_data = load(path)
        self.json_data = AltDataWrapper.pkl_to_json(self.pkl_data)

    def cat(self):
        json_meta = self.json_data
        if isinstance(json_meta, list):
            for target in json_meta:
                print(target)
        else:
            print(json_meta)

    def lookup(self, key, timestamp):
        if key == "":
            self.cat()
            return

        out = DictWrapper(self.json_data).find(key)

        if timestamp == "":
            print("type: {}".format(type(out)))
            print("---------------------")
            print(out)
        else:
            if not isinstance(out, list):
                raise NotImplementedError("{}, should be list".format(key))

            for target in out:
                if str(timestamp) in json.dumps(target):
                    print("---------------------")
                    print(target)


def pkl2json(pkl_meta):
    if isinstance(pkl_meta, str):
        pkl_meta = load(pkl_meta)
    json_data = AltDataWrapper.pkl_to_json(pkl_meta)
    return json_data
