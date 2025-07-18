import numpy as np
from alt.utils.petrel_helper import global_petrel_helper
from loguru import logger


class SimpleLocationDecoder(object):
    def __init__(self, location_path, timestamp_offset=0):
        super().__init__()
        self.location_path = location_path
        self.timestamps = []
        self.location_index = 0
        self.timestamp_offset = timestamp_offset
        self._plugin_init()

    def _plugin_init(self):
        def get_timestamp(line):
            line = line.split(' ')
            return int(line[1])
        self.locations = [i for i in global_petrel_helper.readlines(self.location_path)]
        self.timestamps = [get_timestamp(item) for item in self.locations]
        if self.timestamp_offset != 0:
            logger.warning('{}: add timestamps_offset: {}'.format(self.location_path, self.timestamp_offset))
            self.timestamps = [item + self.timestamp_offset for item in self.timestamps]
        self.timestamps_np = np.array(self.timestamps)

    @property
    def length(self):
        return len(self.timestamps)

    def format(self, data):
        data = data.split(' ')
        one_location = {
            'timestamp': int(data[1]),
            'location': list(map(float, data[2:5])),
            'speed': list(map(float, data[5:8])),
            'yaw': float(data[8])
        }
        return one_location

    def read(self, after=0):
        while True:
            if self.location_index >= self.length:
                return False, None
            timestamp = self.timestamps[self.location_index]
            if timestamp < after:
                self.location_index += 1
                continue
            data = self.locations[self.location_index]
            self.location_index += 1

            one_location = self.format(data)
            return True, one_location

    def search(self, timestamp):
        ind = np.abs(self.timestamps_np - timestamp).argmin()
        one_location = self.format(self.locations[ind])
        time_error = abs(self.timestamps[ind] - timestamp)
        return time_error, one_location


class NearestLocation(object):
    def __init__(self, location_path, timestamp_offset=0) -> None:
        self.location_decoder = SimpleLocationDecoder(location_path, timestamp_offset=timestamp_offset)
        self.timestamps = self.location_decoder.timestamps
        self.timestamp_cache = []
        self.location_cache = []
        self.frame_index = -1
        self.read()

    def read(self):
        flag, one_location = self.location_decoder.read()
        if not flag:
            return None
        else:
            self.frame_index += 1
            self.timestamp_cache.append(self.timestamps[self.frame_index])
            self.location_cache.append(one_location)
            # if len(self.location_cache) > 2:
            #     self.location_cache = self.location_cache[-2:]
            return one_location

    @property
    def last_timestamp(self):
        return self.timestamp_cache[-2]

    @property
    def current_timestamp(self):
        return self.timestamp_cache[-1]

    def between_time(self, timestamp):
        if len(self.timestamp_cache) >= 2:
            if timestamp <= self.timestamp_cache[-1] and timestamp >= self.timestamp_cache[-2]:
                return True
        return False

    def get_nearest_frame(self, timestamp):
        if abs(timestamp - self.timestamp_cache[-1]) < abs(timestamp - self.timestamp_cache[-2]):
            idx = -1
        else:
            idx = -2
        return self.location_cache[idx], abs(timestamp - self.timestamp_cache[idx]), self.timestamp_cache[idx]

    def get_time_gap(self, timestamp):
        return abs(self.current_timestamp - timestamp)

    def __call__(self, timestamp):
        if timestamp < self.timestamps[0]:
            return self.location_cache[0], abs(self.timestamps[0] - timestamp), self.timestamps[0]
        if self.between_time(timestamp):
            return self.get_nearest_frame(timestamp)
        while True:
            one_location = self.read()
            if one_location is None:
                return None, None, None
            if self.get_time_gap(timestamp) < 5*1000*1000:  # 5ms
                return one_location, self.get_time_gap(timestamp), self.current_timestamp
            if self.between_time(timestamp):
                return self.get_nearest_frame(timestamp)


if __name__ == '__main__':
    location_path = '/workspace/kongzelong2/pillar_test/2405/try_merge_pcd/location.tmp.txt'

    location_decoder = SimpleLocationDecoder(location_path)
    print(location_decoder.search(1710988107028000000)[1])

    while True:
        flag, data = location_decoder.read()
        if not flag:
            break

    nearest_decoder = NearestLocation(location_path)
    print(nearest_decoder(1710988107009000000))
    print(nearest_decoder(1710988107083000000))
