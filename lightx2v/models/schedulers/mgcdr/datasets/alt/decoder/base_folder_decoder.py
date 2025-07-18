# Standard Library
import os

# Import from third library
import numpy as np

# Import from alt
from alt.files import smart_listdir


class BaseFolderDecoder(object):
    def __init__(self, path, ext=".jpg", unit="ns"):
        self.path = path
        self.ext = ext

        self.unit = unit
        self.build()

    def build(self):
        metas = {}
        for item in smart_listdir(self.path):
            if not item.endswith(self.ext):
                continue

            if self.unit == "ns":
                timestamp_ns = int(item.split(".")[0])
                timestamp_ms = timestamp_ns // 1000 // 1000
            else:
                timestamp_ms = int(item.split(".")[0])
            metas[timestamp_ms] = os.path.join(self.path, item)

        self.metas = {k: metas[k] for k in sorted(metas)}
        self.timestamps = list(self.metas.keys())

    def get_nearst(self, timestamp_ms):
        timestamp_gaps = np.abs(np.array(self.timestamps) - timestamp_ms)
        select_index = timestamp_gaps.argmin()
        return self.metas[self.timestamps[select_index]], timestamp_gaps[select_index]

    def get(self, timestamp_ms):
        return self.metas[timestamp_ms]

    def remove(self, timestamp_ms):
        assert timestamp_ms in self.timestamps
        self.timestamps.remove(timestamp_ms)
        self.metas.pop(timestamp_ms)
        assert timestamp_ms not in self.metas

    def add(self, timestamp_ms, value):
        assert timestamp_ms not in self.timestamps
        assert timestamp_ms not in self.metas
        self.metas[timestamp_ms] = value
        self.timestamps.append(timestamp_ms)
        self.timestamps.sort()
        self.metas = {k: self.metas[k] for k in sorted(self.metas)}
