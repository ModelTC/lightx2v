import bisect
import os
from datetime import datetime
from decimal import Decimal
import dateutil.tz
import addict
import numpy as np
from loguru import logger

from alt.decoder.utils.misc import parse_decimal, update_kwargs
from alt.files import smart_exists, smart_listdir, smart_isdir
from alt.utils.petrel_helper import global_petrel_helper


class SensorDecoder(object):
    def __init__(self,
                 sensor_type,
                 sensor_name,
                 datapath,
                 meta=None,
                 cache=None,
                 calib_mode=None, # #HEAR
                 source_sensor=None, # #HEAR
                 target_sensor='ground_projection',
                 target_system='FLU',
                 timpath=None,
                 tim_eps=1e-4,
                 maxlen=None,
                 timsep=', ',
                 cal_len=True,
                 cal_freq=True,
                 iter_kwargs=None,
                 loader=None,
                 **kwargs):
        self.sensor_type = sensor_type
        self.sensor_name = sensor_name
        self.datapath = datapath
        
        self.meta = meta
        self.cache = cache
        self.cache.calib_mode = calib_mode

        self.source_sensor = sensor_name if source_sensor is None else source_sensor
        self.target_sensor, self.target_system = target_sensor, target_system

        self.param = self.load_calib_param(sensor_name=self.source_sensor)
        
        if timpath is not None:
            assert not hasattr(self, 'timestamps'), 'timestamps have been set'
            self.timestamps = self.load_timestamps(timpath, maxlen, timsep, loader)
        else:
            assert getattr(self, 'timestamps', None) is not None, 'timestamps are required'
        assert all(ts1 < ts2 for ts1, ts2 in zip(self.timestamps[:-1], self.timestamps[1:])), 'disordered timestamps'
        self.tim_eps = tim_eps

        if cal_len:
            self.seq_len = len(self)
        if cal_freq:
            self.seq_freq = self.calculate_frequency(self.timestamps, self.tim_eps)

        self._iter_kwargs, self.iter_kwargs = update_kwargs(iter_kwargs), None
        self.update_iter()
    
    def parse_intrinsics(self, param):
        """Parse intrinsics parameters"""
        intrinsics = addict.Dict()
        assert param['sensor_name'] == param['target_sensor_name'], 'invalid intrinsics file'
        if param['device_type'] != self.sensor_type:
            logger.warn('mismatched device type for intrinsics', RuntimeWarning)
            return None
        for param_k, param_v in param['param'].items():
            if not isinstance(param_v, dict) or param_v.get('data', None) is None:
                setattr(intrinsics, param_k, param_v)
                continue
            setattr(intrinsics, param_k, np.array(param_v['data']).reshape((param_v['rows'], param_v['cols'])))
        return intrinsics

    def parse_extrinsics(self, param):
        """Parse extrinsics parameters"""
        if param['target_sensor_name'] != self.target_sensor:
            root = self.load_calib_param(sensor_name=param['target_sensor_name'], skip_intrinsics=True, skip_hmatrix=True).extrinsics
        else:
            root = np.eye(4)

        assert param['device_type'] == 'relational', 'invalid device type'
        sensor_calib = param['param']['sensor_calib']
        extrinsics = np.array(sensor_calib['data']).reshape((sensor_calib['rows'], sensor_calib['cols']))
        return root @ extrinsics

    def parse_hmatrix(self, param):
        """Parse homography matrix"""
        assert param['sensor_name'] == param['target_sensor_name'], 'invalid homography file'
        assert param['device_type'] == self.sensor_type, 'invalid device type'
        param = param['param']['h_matrix']
        hmatrix = np.array(param['data']).reshape((param['rows'], param['cols']))
        return hmatrix
        
    def load_calib_param(self, sensor_name, skip_intrinsics=False, skip_hmatrix=False):
        """Load calibration parameters"""
        sensor_cfg = os.path.join(self.datapath.cfgdir, sensor_name)
        assert smart_exists(sensor_cfg), f'sensor config does not exist: {sensor_name}'

        intrinsics, extrinsics, hmatrix = [None] * 3
        for filename in sorted(smart_listdir(sensor_cfg)):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(sensor_cfg, filename)
            param = global_petrel_helper.load_json(filepath)
            assert len(param) == 1, 'invalid parameter format'
            param = next(iter(param.values()))
            assert param['sensor_name'] == sensor_name, 'invalid parameter contained'
            if param['param_type'] == 'intrinsic':
                if skip_intrinsics:
                    continue
                intrinsics = self.parse_intrinsics(param)
            elif param['param_type'] == 'extrinsic':
                if extrinsics is not None:
                    continue
                extrinsics = self.parse_extrinsics(param)
            elif param['param_type'] == 'h_matrix':
                if skip_hmatrix:
                    continue
                hmatrix = self.parse_hmatrix(param)
        param = addict.Dict()
        param.update(intrinsics=intrinsics, extrinsics=extrinsics, hmatrix=hmatrix)
        return param
    
    def update_iter(self, kwargs=None):
        """Update iter kwargs"""
        self.iter_kwargs = update_kwargs(self._iter_kwargs, kwargs)
        # item_idx is required
        self.iter_kwargs['item_idx'] = self.fetch_idx(self.iter_kwargs.pop('item_idx', None))
    
    @staticmethod
    def convert_timestamp(timestamp, timezone='Asia/Shanghai', digit=16, point=3):
        """Convert format of timestamp"""
        if isinstance(timestamp, Decimal):
            decimal = timestamp.as_tuple()
            assert len(decimal.digits) == digit + \
                point and decimal.exponent == -point, 'invalid decimal'
            return timestamp
        # format conversion
        if isinstance(timestamp, (int, float)):
            timestamp = str(timestamp)
        if isinstance(timestamp, str):
            _timestamp = timestamp.replace('_', '', 1)
            if not _timestamp.isnumeric():
                _timestamp = timestamp.replace('.', '')
                if not _timestamp.isnumeric():
                    if timezone is None or isinstance(timezone, str):
                        timezone = dateutil.tz.gettz(timezone)
                    elif not isinstance(timezone, dateutil.zoneinfo.tzfile):
                        raise ValueError('invalid timezone')
                    _timestamp = str(datetime.strptime(
                        timestamp, r'%Y-%m-%d-%H-%M-%S-%f').replace(tzinfo=timezone).timestamp())
            timestamp = _timestamp
        else:
            raise TypeError('invalid timestamp')
        # unit conversion
        decimal = addict.Dict(parse_decimal(timestamp))
        scale = digit - decimal.digit
        timestamp = Decimal(timestamp) * Decimal(str(10 ** scale))
        if point is None:
            return timestamp
        scale = point - decimal.point
        if scale == 0:
            return timestamp
        if scale > 0:
            return timestamp + Decimal('.' + '0' * point)
        return Decimal(str(timestamp)[:scale])

    def load_timestamps(self, timpath, maxlen=None, timsep=', ', loader=None):
        """Sequence timestamps"""
        ts_conts = self.fetch_loader(loader).get(
            timpath, return_obj=True).rstrip().rsplit('\n')
        item = ts_conts[0].split(timsep)
        ele_num = len(item)
        if ele_num == 1:
            timestamps = list(map(self.convert_timestamp, ts_conts))
        elif ele_num == 2:
            timestamps, idxoff = list(), int(item[0]) - 1
            for ts_cont in ts_conts:
                idx, t_str = ts_cont.split(timsep)
                idx = int(idx) - idxoff - 1 - len(timestamps)
                if idx == 0:
                    timestamps.append(self.convert_timestamp(t_str))
                elif idx < 0:
                    timestamps[idx] = self.convert_timestamp(t_str)
                else:
                    raise ValueError('invalid timestamp index')
        else:
            raise NotImplementedError
        if maxlen is None:
            return timestamps
        return timestamps[:min(maxlen, len(timestamps))]

    @staticmethod
    def calculate_frequency(timestamps, epsilon=1e-4, sec_digit=10):
        """Calculate sequence frequency by timestamp"""
        assert len(timestamps) > 1, 'not enough timestamps to calculate frequency'
        decimal = timestamps[0].as_tuple()
        digit = len(decimal.digits) + decimal.exponent
        scale = digit - sec_digit
        ts_arr = np.array(timestamps)  # decimal array
        interval = np.median(ts_arr[1:] - ts_arr[:-1])
        freq = 10 ** scale / interval
        if freq < 1.:
            inverse = 1 / freq
            ideal = round(inverse)
            if float(abs(inverse - ideal)) < epsilon:
                inverse = ideal
            freq = 1 / inverse
        else:
            ideal = round(freq)
            if float(abs(freq - ideal)) < epsilon:
                freq = ideal
        return float(freq)
    
    def __len__(self):
        # return getattr(self, 'seq_len', len(self.timestamps)) - 1  # ERROR
        return len(self.timestamps) - 1
    
    
    def forward_items(self, item_idx, **kwargs):
        """Forward items"""
        raise NotImplementedError

    def format_item(self, item, item_idx, timestamp=None):
        """Format item information"""
        if timestamp is None:
            timestamp = self.timestamps[item_idx]
        return addict.Dict(data=item, idx=item_idx, ts=timestamp)

    def fetch_idx(self, item_idx=None):
        """Fetch item idx"""
        if item_idx is None:
            item_idx = self.iter_kwargs.get('item_idx', -1)
        assert isinstance(item_idx, int), 'invalid type of item index'
        # print(item_idx)
        assert -1 <= item_idx < len(self.timestamps), 'invalid value of item index'
        return item_idx

    def fetch_step(self, sample_step=None):
        """Fetch sample step"""
        if sample_step is None:
            sample_step = self.iter_kwargs.get('sample_step', 1)
        assert isinstance(sample_step, int), 'invalid type of sample step'
        return sample_step

    def __getitem__(self, item_idx, **kwargs):
        """Fetch a specific item"""
        assert item_idx > -1, 'invalid item index to get'
        items = self.forward_items(item_idx=item_idx, **kwargs)
        item = next(items)
        return self.format_item(item, item_idx=item_idx)
    
    def get_item(self):
        raise NotImplementedError
    
    def retrieve_item(self, timestamp, timestamps=None, mode=None, max_interval=None,
                      only_index=False, return_weight=False):
        """Retrieve item by timestamp"""
        if timestamps is None:
            timestamps = self.timestamps
        else:
            assert all(ts1 < ts2 for ts1, ts2 in zip(
                timestamps[:-1], timestamps[1:])), 'disordered timestamps'
            assert only_index, 'item cannot be returned for custom timestamps'
        # tim_errors = np.array(timestamps) - timestamp
        # tim_intervals = np.abs(tim_errors)
        # near_idx = np.argpartition(tim_intervals, 2)[:2]
        # if np.sign(tim_errors[near_idx].prod()) > 0:
        #     near_idx = [near_idx[np.argmin(tim_intervals[near_idx])]]

        index = bisect.bisect_left(timestamps, timestamp)
        if index < len(timestamps) and timestamps[index] == timestamp:
            indices = [index]
        else:
            indices = list(filter(
                lambda idx: -1 < idx < len(timestamps), [index - 1, index]))
            if mode == 'nearest' and len(indices) > 1:
                indices = [indices[0] if timestamp - timestamps[indices[0]]
                           < timestamps[indices[1]] - timestamp else indices[1]]
            if max_interval is not None:
                indices = list(filter(lambda idx: abs(
                    timestamps[idx] - timestamp) < max_interval, indices))
        if not return_weight:
            return indices if only_index else [self[idx] for idx in indices]
        if len(indices) == 0:
            weights = list()
        elif len(indices) == 1:
            weights = [1.]
        else:
            interval = timestamps[indices[1]] - timestamps[indices[0]]
            weights = [1 - float(abs(timestamps[idx] - timestamp) / interval)
                       for idx in indices]
        return indices if only_index else [self[idx] for idx in indices], weights

    def items(self, sample_step=1, **kwargs):
        for item_idx in range(0, len(self), sample_step):
            item = self.get_item(item_idx)
            yield self.format_item(item, item_idx=item_idx)
