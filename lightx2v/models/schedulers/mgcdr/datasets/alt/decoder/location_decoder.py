"""Location Info Management"""

import copy
import os
from collections import OrderedDict, defaultdict

import addict
import numpy as np
import pandas as pd
import pymap3d
import scipy.interpolate
from scipy.spatial.transform import Rotation


from alt.decoder.base_decoder import SensorDecoder
from alt.decoder.utils.misc import update_kwargs, normalize_rad
from alt import smart_exists, smart_isdir, smart_listdir
from alt.utils.petrel_helper import global_petrel_helper


class LocationDecoder(SensorDecoder):
    """Location Module"""
    INLOC_EXTENSIONS = ['.tmp.txt', '.csv']
    DEFAULT_ATTRS = addict.Dict(
        pat='pat', ts='ts', pos='pos', vel='vel', rot='rot', avel=None, acc=None,
        sep='.', dig=None, throt=None, brake=None, steer=None, conf=None)
    LOC_DIMS = ('x', 'y', 'z')
    LOC_NDIM = len(LOC_DIMS)
    ANG_NAME, ANG_DIMS = 'euler', ('r', 'p', 'y')
    AVAILABLE_ROTS = {
        ANG_NAME: len(ANG_DIMS), 'rotvec': 3, 'quat': 4, 'matrix': 9}
    AVAILABLE_PARENTS = dict(WGS84='geodetic', GCJ02='gcj', BD09='bd')
    ORG_DIMS = ('lat', 'lon', 'h')
    ORG_MAPPER = {
        'HZ': (30.895136130454, 121.932467792949, 0.),
        'SH': (30.89513613045363, 121.93246779294897, 0.),
        'WX': (31.5917804561138, 120.413429344334, 0.)}
    ORG_NDIM = len(ORG_DIMS)
    ORG_BEV = (ORG_DIMS.index('lat'), ORG_DIMS.index('lon'))
    LOC_PATTERN = '[OdometryInfo]'

    
    def __init__(self, datapath, sensor_name=None, loc_attrs=None, source_parent=None,
                 source_system=None, origin=None, region=None, cal_motion=False,
                 intrpl_degree=None, access_ele=False, cal_rotmat=False, loader=None, **kwargs):
        
        self.locpath, self.cfgdir, self.vehtype = datapath.get('path'), datapath.get('config'), datapath.get('vehicle')
        assert smart_exists(self.locpath), 'location path does not exist'
        
        
        locstem, loc_ext = os.path.splitext(os.path.basename(self.locpath))
        assert loc_ext in self.INLOC_EXTENSIONS, loc_ext
        
        sensor_name = locstem if sensor_name is None else sensor_name

        vehicle = self.parse_vehinfo(os.path.dirname(self.locpath), self.cfgdir, self.vehtype)
        datapath = addict.Dict(locpath=self.locpath, cfgdir=vehicle.cfgdir, vehtype=vehicle.form)
        self.loc_attrs = addict.Dict(update_kwargs(self.DEFAULT_ATTRS, loc_attrs, keepattr=True))

        meta, cache = self.parse_locinfo(self.locpath, loc_ext, self.loc_attrs)
        cache.update(self.parse_gloinfo(source_parent, source_system, origin, region))

        self.cal_motion, self.access_ele, self.cal_rotmat = cal_motion, access_ele, cal_rotmat
        self.locations, self.timestamps, self.supplements = self.load_locations(self.locpath, meta, cache, loader)
        assert len(self) > 0, 'failed to load location list'
        self.tck = self.calculate_spline(self.locations, self.timestamps, intrpl_degree)

        super().__init__(sensor_type='location', sensor_name=sensor_name, datapath=datapath, meta=meta, cache=cache, **kwargs)

    
    @classmethod
    def parse_locinfo(cls, locpath, loc_ext, loc_attrs):
        """Parse location information"""
        if loc_ext is None:
            assert locpath is not None, 'location path is required if extension is missing'
            loc_ext = os.path.splitext(locpath)[-1]
        if isinstance(loc_attrs.ts, str):
            loc_attrs.ts = [loc_attrs.ts]
        elif not isinstance(loc_attrs.ts, (list, tuple)):
            raise TypeError('invalid timestamp name for location')
        if isinstance(loc_attrs.dig, (int, type(None))):
            loc_attrs.dig = [loc_attrs.dig] * len(loc_attrs.ts)
        elif isinstance(loc_attrs.dig, (list, tuple)):
            assert len(loc_attrs.ts) == len(loc_attrs.dig), \
                'mismatch between name and digit of timestamp'
        else:
            raise TypeError('invalid timestamp digit for location')
        loc_ios = dict(dtype=defaultdict(
            lambda: np.dtype('float64'), dict.fromkeys([loc_attrs.pat, *loc_attrs.ts], str)))
        pref = f'{loc_attrs.rot}{loc_attrs.sep}'
        if locpath is None or loc_ext == '.tmp.txt':
            rot_mode, rot_dims = cls.ANG_NAME, [
                cls.ANG_DIMS[2]]  # default rotation
            loc_rot = [f'{pref}{dim}' for dim in rot_dims]
            columns = [loc_attrs.pat, *loc_attrs.ts]
            for attr in ['pos', 'vel']:
                pref = f'{loc_attrs[attr]}{loc_attrs.sep}'
                columns.extend([f'{pref}{dim}' for dim in cls.LOC_DIMS])
            columns.extend(loc_rot)
            loc_ios.update(delimiter=' ', header=None, names=columns)
        elif loc_ext == '.csv':
            columns = pd.read_csv(locpath, index_col=0, nrows=0).columns
            raw_dims, loc_rot = list(), list()
            for col in columns:
                if not col.startswith(pref):
                    continue
                dim = col.split(loc_attrs.sep, 1)[1]
                assert dim not in raw_dims, 'repeated rotation dimension'
                _ = raw_dims.append(dim), loc_rot.append(col)
            for rot_mode, ndim in cls.AVAILABLE_ROTS.items():
                if rot_mode == cls.ANG_NAME:
                    if set(raw_dims).issubset(cls.ANG_DIMS):
                        rot_dims = raw_dims
                        break
                elif len(raw_dims) == ndim:
                    rot_dims = cls.ANG_DIMS
                    break
            else:
                raise RuntimeError('invalid rotation mode')
        else:
            raise ValueError('invalid location file')
        rot_axes = [cls.ANG_DIMS.index(dim) for dim in rot_dims]
        assert all(r1 < r2 for r1, r2 in zip(rot_axes[:-1], rot_axes[1:])), \
            'disordered rotation axes'
        meta = addict.Dict(rot_mode=rot_mode, rot_dims=rot_dims,
                           rot_axes=rot_axes, rot_ndim=len(rot_dims))
        cache = addict.Dict(
            loc_ios=loc_ios, loc_attrs=loc_attrs, loc_rot=loc_rot)
        return meta, cache

    @classmethod
    def parse_gloinfo(cls, parent, child, origin, region=None):
        """Parse global information"""
        assert parent is None or parent in cls.AVAILABLE_PARENTS, \
            'invalid parent coordinate system'
        assert child is None or child in cls.AVAILABLE_SYSTEMS, \
            'invalid child coordinate system'
        if origin is not None:
            assert region is None, 'at most one is required between origin and region'
            if isinstance(origin, dict):
                _origin = list()
                for dim in cls.ORG_DIMS:
                    assert dim in origin, 'incomplete enu origin'
                    _origin.append(origin[dim])
                origin = _origin
            elif not isinstance(origin, (tuple, list)):
                raise TypeError('invalid enu origin')
        elif region is not None:
            assert region in cls.ORG_MAPPER, 'invalid enu region'
            origin = cls.ORG_MAPPER[region]
        return addict.Dict(parent=parent, child=child, origin=origin)

    @classmethod
    def convert_rotation(cls, rot, src_mode=None, tgt_mode=None):
        """Convert rotation"""
        assert rot.ndim == 2, 'invalid rotation format'
        if rot.shape[1] < cls.AVAILABLE_ROTS[cls.ANG_NAME]:  # incomplete angles
            return normalize_rad(rot)
        if src_mode is None:
            src_mode = cls.ANG_NAME
        else:
            assert src_mode in cls.AVAILABLE_ROTS, 'invalid source rotation mode'
        assert rot.shape[1] == cls.AVAILABLE_ROTS[src_mode], \
            'mismatch between rotation data and mode'
        if tgt_mode is None:
            tgt_mode = cls.ANG_NAME
        else:
            assert tgt_mode in cls.AVAILABLE_ROTS, 'invalid target rotation mode'
        if src_mode == tgt_mode:
            return rot
        if src_mode == 'euler':
            rot = Rotation.from_euler('zyx', rot[:, ::-1])
        elif src_mode == 'matrix':
            rot = Rotation.from_matrix(rot.reshape(-1, 3, 3))
        else:
            rot = getattr(Rotation, f'from_{src_mode}')(rot)
        if tgt_mode == 'euler':
            rot = rot.as_euler('zyx')[:, ::-1]
        elif tgt_mode == 'matrix':
            rot = rot.as_matrix().reshape(-1, cls.AVAILABLE_ROTS[tgt_mode])
        else:
            rot = getattr(rot, f'as_{tgt_mode}')()
        return rot

    @classmethod
    def calculate_motion(cls, cur, later=None):
        """Calculate motion"""
        if later is None:
            motion = dict(
                acc=np.zeros(cls.LOC_NDIM), jerk=np.zeros(cls.LOC_NDIM))
            if 'vel' not in cur:
                motion['vel'] = np.zeros(cls.LOC_NDIM)
            cur.update(motion)
            return cur
        delta_ts = float(later.ts - cur.ts)
        if 'vel' not in cur:
            cur.vel = (later.pos - cur.pos) / delta_ts
        acc = (later.vel - cur.vel) / delta_ts
        jerk = (later.acc - acc) / delta_ts
        cur.update(acc=acc, jerk=jerk)
        return cur

    def load_locations(self, locpath, meta, cache, loader=None):
        """Load location"""
        # loader = self.fetch_loader(loader)
        data_frame = global_petrel_helper.read_csv(locpath, **cache.loc_ios)

        for attr, dig in zip(cache.loc_attrs.ts, cache.loc_attrs.dig):
            if dig is None:
                continue
            data_frame[attr] = data_frame[attr].str.zfill(dig)
        timestamps = data_frame[cache.loc_attrs.ts].agg(''.join, axis=1).map(self.convert_timestamp)
        locs, supplements = dict(), dict()
        for attr in ['pos', 'vel', 'avel', 'acc']:
            if cache.loc_attrs[attr] is None:
                continue
            pref = f'{cache.loc_attrs[attr]}{cache.loc_attrs.sep}'
            locs[attr] = data_frame[[f'{pref}{dim}' for dim in self.LOC_DIMS]].to_numpy()
        locs['rot'] = self.convert_rotation(data_frame[cache.loc_rot].to_numpy(), src_mode=meta.rot_mode)
        for attr in ['throt', 'brake', 'steer']:
            if cache.loc_attrs[attr] is not None:
                supplements[attr] = data_frame[cache.loc_attrs[attr]].to_numpy()
        locations = OrderedDict()
        for idx, timestamp in enumerate(timestamps):
            location = addict.Dict({item[0]: item[1][idx] for item in locs.items()})
            location.update(ts=timestamp)
            locations[timestamp] = location
        timestamps, locations = zip(*locations.items())
        if self.cal_motion:
            for cur, later in zip(locations[::-1], reversed([*locations[1:], None])):
                cur.update(self.calculate_motion(cur, later))
        _ = [loc.pop('ts') for loc in locations]
        return locations, timestamps, supplements

    @classmethod
    def calculate_spline(cls, locations, timestamps, intrpl_degree=None):
        """Calculate spline"""
        if intrpl_degree is None:
            return None
        assert 0 < intrpl_degree < 6, 'invalid interpolation degree'
        if len(locations) < intrpl_degree + 1:
            return None
        attrs, timestamps = locations[0].keys(), np.array(timestamps)
        delta_ts = (timestamps - timestamps[0]).astype(float)
        tcks = dict()
        for attr in attrs:
            arr = np.stack([location[attr] for location in locations])
            if attr == 'rot':
                arr = cls.convert_rotation(arr, tgt_mode='quat')
            tcks[attr] = scipy.interpolate.splprep(
                arr.transpose(), u=delta_ts, k=intrpl_degree, s=0)[0]
        return tcks

    def load_meta_info(self, meta):
        meta = super().load_meta_info(meta)
        h_axis = self.LOC_DIMS.index(
            self.AVAILABLE_SYSTEMS[self.target_system].h)
        bev_axes = modulo_axis(h_axis, self.LOC_NDIM)
        rot_hidx = meta.rot_axes.index(h_axis)
        meta.update(h_axis=h_axis, bev_axes=bev_axes, rot_hidx=rot_hidx)
        return meta

    def load_cache_data(self, cache, calib_mode, **kwargs):
        if calib_mode is None:  # default mode
            calib_mode = self.sensor_name
        if calib_mode == 'global':
            assert cache.parent is not None and cache.child is not None and \
                cache.origin is not None, \
                'parent, child, and origin are required for global location'
        elif calib_mode != 'local':
            raise ValueError('invalid calib mode')
        return super().load_cache_data(cache, calib_mode, **kwargs)

    def parse_vehinfo(self, datadir, vehcfg=None, vehtype=None):
        """Parse vehicle information"""
        cfgroot = os.path.join(datadir, 'config/vehicle')
        if vehtype is None:
            cfgroot = cfgroot if vehcfg is None else os.path.dirname(vehcfg.rstrip('/'))
            vehtypes = smart_listdir(cfgroot)
            assert len(vehtypes) == 1, 'no enough or too many vehicle types detected'
            vehtype = vehtypes[0]
        if vehcfg is None:
            vehcfg = os.path.join(cfgroot, vehtype)
        return addict.Dict(cfgdir=vehcfg, form=vehtype)
    
    
    @classmethod
    def parent2child(cls, arr, cache):
        """Transform coordinate from parent to child"""
        assert arr.ndim == 2 and arr.shape[1] == cls.ORG_NDIM, \
            'invalid coordinate array'
        tf_func = getattr(UnifiedMapper, f'{cls.AVAILABLE_PARENTS[cache.parent]}2geodetic', None)
        assert tf_func is not None, 'invalid parent coordinate system'
        arr[:, cls.ORG_BEV[0]], arr[:, cls.ORG_BEV[1]] = tf_func(
            *arr[:, cls.ORG_BEV].transpose())
        assert len(cache.origin) == cls.ORG_NDIM, 'invalid enu origin'
        arr = np.stack(pymap3d.geodetic2enu(
            *arr.transpose(), *cache.origin)).transpose()
        if cache.child == 'FLU':
            arr[:, [0, 1]] = arr[:, [1, 0]]
            arr[:, 1] = -arr[:, 1]
        elif cache.child == 'LFD':
            arr[:, [0, 2]] = -arr[:, [0, 2]]
        elif cache.child == 'FRD':
            arr[:, [0, 1]] = arr[:, [1, 0]]
            arr[:, 2] = -arr[:, 2]
        elif cache.child != 'RFU':
            raise ValueError('invalid child coordinate system')
        return arr

    @classmethod
    def child2parent(cls, arr, cache):
        """Transform coordinate from child to parent"""
        assert arr.ndim == 2 and arr.shape[1] == cls.LOC_NDIM, \
            'invalid coordinate array'
        if cache.child == 'FLU':
            arr[:, [0, 1]] = arr[:, [1, 0]]
            arr[:, 0] = -arr[:, 0]
        elif cache.child == 'LFD':
            arr[:, [0, 2]] = -arr[:, [0, 2]]
        elif cache.child == 'FRD':
            arr[:, [0, 1]] = arr[:, [1, 0]]
            arr[:, 2] = -arr[:, 2]
        elif cache.child != 'RFU':
            raise ValueError('invalid child coordinate system')
        assert len(cache.origin) == cls.ORG_NDIM, 'invalid enu origin'
        arr = np.stack(pymap3d.enu2geodetic(
            *arr.transpose(), *cache.origin)).transpose()
        tf_func = getattr(UnifiedMapper, f'geodetic2{cls.AVAILABLE_PARENTS[cache.parent]}', None)
        assert tf_func is not None, 'invalid parent coordinate system'
        arr[:, cls.ORG_BEV[0]], arr[:, cls.ORG_BEV[1]] = tf_func(
            *arr[:, cls.ORG_BEV].transpose())
        return arr

    @classmethod
    def calculate_rotmat(cls, meta, angs, dims=None):
        """Calculate rotation matrix"""
        assert angs.shape == (meta.rot_ndim, ), 'invalid rotation angles'
        if dims is None:
            _angs = np.zeros(cls.AVAILABLE_ROTS[cls.ANG_NAME])
            _angs[meta.rot_axes] = angs
            rmat = cls.convert_rotation(
                _angs[None, ...], tgt_mode='matrix').reshape(3, 3)
            return rmat
        angs = angs[[meta.rot_hidx], None]
        rmat = radian2matrix(angs, axes={2: None, 3: meta.bev_axes}[dims])
        return rmat.squeeze(0)

    @classmethod
    def calculate_rotang(cls, meta, rmat):
        """"Calculate rotation angles"""
        assert rmat.shape == (3, 3), 'invalid rotation matrix'
        _angs = cls.convert_rotation(
            rmat.reshape(1, -1), src_mode='matrix').squeeze(0)
        angs = _angs[meta.rot_axes]
        return angs

    def transform_location(self, loc):
        """Transform single location"""
        assert 'calib_mode' in self.cache, 'calib mode not registered yet'
        # transform from global to local coordinate
        if self.cache.calib_mode == 'global':
            loc.pos = self.parent2child(loc.pos[None, ...], self.cache).squeeze(0)
        # transform from source to target coordinate
        if self.param.intrinsics is not None:
            self.log('location intrinsics are ignored', 'warn')
        rmat = self.calculate_rotmat(self.meta, loc.pop('rot'))
        ext_mat = np.linalg.inv(self.param.extrinsics)
        pos = (rmat @ ext_mat[:-1, -1:]).squeeze(-1) + loc.pop('pos')
        rmat = rmat @ ext_mat[:-1, :-1]
        rot = self.calculate_rotang(self.meta, rmat)
        for attr, val in loc.items():
            loc[attr] = (ext_mat[:-1, :-1] @ val[..., None]).squeeze(-1)
        loc.pos, loc.rot = pos, rot
        # add extra entrance to access element
        if self.access_ele:
            for attr, val in loc.items():
                dims = self.meta.rot_dims if attr == 'rot' else self.LOC_DIMS
                loc[attr] = dict(arr=val, **dict(zip(dims, val)))
            if self.cal_rotmat:
                loc.rot['mat'] = rmat
        elif self.cal_rotmat:
            loc.rot = dict(arr=loc.rot, mat=rmat)
        return loc

    def forward_items(self, item_idx, sample_step=None, **kwargs):
        item_idx, sample_step = self.fetch_idx(item_idx), self.fetch_step(sample_step)
        assert not kwargs, f'unrecognized arguments are contained: {kwargs}'
        if item_idx > -1:
            location = copy.deepcopy(self.locations[item_idx])
            yield self.transform_location(location)
        else:
            for item_idx in range(0, len(self), sample_step):
                location = copy.deepcopy(self.locations[item_idx])
                yield self.transform_location(location)
    
    def get_item(self, item_idx):
        location = copy.deepcopy(self.locations[item_idx])
        return self.transform_location(location)

    def format_item(self, item, item_idx, timestamp=None):
        item = super().format_item(item, item_idx, timestamp)
        if self.iter_kwargs.get('save_mode', None) is None:
            item.meta = self.meta
        return item

    def retrieve_item(self, timestamp, timestamps=None, mode=None, max_interval=None,
                      only_index=False, return_weight=False):
        if mode != 'fused':
            return super().retrieve_item(timestamp, timestamps, mode, max_interval, only_index, return_weight)
        if only_index:
            indices, weights = super().retrieve_item(timestamp, timestamps, None, max_interval, True, True)
            if len(indices) == 0:
                return (indices, weights) if return_weight else indices
            item_idx = sum(index * weight for index, weight in zip(indices, weights))
            return ([item_idx], [1.]) if return_weight else [item_idx]
        assert timestamps is None, 'item cannot be returned for custom timestamps'
        if self.tck is None:
            indices, weights = super().retrieve_item(timestamp, None, None, max_interval, True, True)
            if len(indices) == 0:
                return (indices, weights) if return_weight else indices
            location = addict.Dict(dict.fromkeys(self.locations[0], 0.))
            for index, weight in zip(indices, weights):
                for attr, val in self.locations[index].items():
                    location[attr] += val * weight
        else:
            if max_interval is not None:
                if self.timestamps[0] - timestamp > max_interval or timestamp - self.timestamps[-1] > max_interval:
                    return (list(), list()) if return_weight else list()
            delta_ts = float(timestamp - self.timestamps[0])
            location = addict.Dict()
            for attr, tck in self.tck.items():
                arr = np.array(scipy.interpolate.splev(delta_ts, tck, der=0, ext=0))
                if attr == 'rot':
                    arr = self.convert_rotation(arr[None, ...], src_mode='quat').squeeze(0)
                location[attr] = arr
        item = self.format_item(item=self.transform_location(location), item_idx=None, timestamp=timestamp)
        return ([item], [1.]) if return_weight else [item]

    # def write_items(self, item_idx, sample_step, save_mode,
    #                 overwrite=False, skip_exist=False, **kwargs):
    #     assert not kwargs, f'unrecognized arguments are contained: {kwargs}'
    #     loader = self.fetch_loader()
    #     save_mode = 'csv' if save_mode is True else save_mode

    #     def _write_func(_locs, _loc_idx):
    #         outfile = self.fetch_outname(
    #             sample_step=sample_step, extension=save_mode)
    #         if not overwrite and loader.exists(outfile):
    #             if skip_exist:
    #                 return
    #             raise FileExistsError(outfile)
    #         _loc_idx = max(0, _loc_idx)
    #         if save_mode == 'csv':
    #             data_frame = defaultdict(list)
    #             for loc in _locs:
    #                 data_frame['ts'].append(self.timestamps[_loc_idx])
    #                 for attr, val in loc.items():
    #                     pref = f'{attr}{self.cache.loc_attrs.sep}'
    #                     dims = self.meta.rot_dims if attr == 'rot' else self.LOC_DIMS
    #                     if isinstance(val, dict):
    #                         val = val['arr']
    #                     for dim, _val in zip(dims, val):
    #                         data_frame[f'{pref}{dim}'].append(_val)
    #                 for attr, val in self.supplements.items():
    #                     data_frame[attr].append(val[_loc_idx])
    #                 _loc_idx += sample_step
    #             buffer = pd.DataFrame(data_frame).to_csv(
    #                 header=True, index=False)
    #         elif save_mode == 'tmp.txt':
    #             data_list = list()
    #             for loc in _locs:
    #                 data = [self.LOC_PATTERN, self.timestamps[_loc_idx]]
    #                 for attr in ['pos', 'vel', 'rot']:
    #                     val = loc[attr]['arr'] if isinstance(
    #                         loc[attr], dict) else loc[attr]
    #                     data.extend(val)
    #                 data_list.append(' '.join(map(str, data)))
    #                 _loc_idx += sample_step
    #             buffer = '\n'.join(data_list)
    #         else:
    #             raise ValueError('invalid save mode')
    #         loader.put(buffer.encode('utf-8'), outfile,
    #                    overwrite=overwrite, skip_exist=skip_exist)

    #     thread = threading.Thread(
    #         target=_write_func, name=f'{self.ident}.write', args=(self.forward_items(
    #             item_idx=item_idx, sample_step=sample_step), item_idx), daemon=True)
    #     thread.start()
    #     return thread

    @classmethod
    def convert_coordinate(cls, loc, mode, arr=None):
        """Convert coordinate system via change of basis"""
        if isinstance(loc.data.rot, dict):
            rotmat = loc.data.rot.get(
                'mat', cls.calculate_rotmat(loc.meta, loc.data.rot['arr']))
        else:
            rotmat = cls.calculate_rotmat(loc.meta, loc.data.rot)
        pos = loc.data.pos['arr'] if isinstance(
            loc.data.pos, dict) else loc.data.pos
        if arr is None:
            proj_mat = np.eye(4)
            proj_mat[:-1, :-1] = rotmat
            proj_mat[:-1, -1] = pos
            if mode == 'ego2wld':
                return proj_mat
            if mode == 'wld2ego':
                return np.linalg.inv(proj_mat)
            raise ValueError('invalid conversion mode')
        assert arr.ndim == 2, 'invalid coordinate array'
        if mode == 'ego2wld':
            assert arr.shape[1] == rotmat.shape[0], 'mismatch between coordinate and rotation'
            ego_arr = arr[..., None]
            wld_arr = (rotmat[None, ...] @
                       ego_arr).squeeze(-1) + pos[None, ...]
            return wld_arr
        if mode == 'wld2ego':
            assert arr.shape[1] == rotmat.shape[1], 'mismatch between coordinate and rotation'
            wld_arr = arr[..., None]
            ego_arr = (np.linalg.inv(rotmat)[
                       None, ...] @ (wld_arr - pos[None, :, None])).squeeze(-1)
            return ego_arr
        raise ValueError('invalid conversion mode')


if __name__ == '__main__':
    LOC_PATH = {'path': 'ad_system_common:s3://sdc_gac/Data_Collection/GT_data/gacGtParser/drive_gt_collection/PVB_gt/2024-04/A02-290/2024-04-07/2024_04_07_00_39_44_gacGtParser/ved/pose/location.tmp.txt',
                'config': 'ad_system_common:s3://sdc_gac/Data_Collection/GT_data/gacGtParser/drive_gt_collection/PVB_gt/2024-04/A02-290/2024-04-07/2024_04_07_00_39_44_gacGtParser/calib',
                'vehicle': 'A02-290'}
    
    # module = LocationDecoder(
    #     datapath=LOC_PATH, sensor_name='local', source_sensor='car_center',
    #     outtag='2022_07_13_11_11_06_AutoCollect')
    # # __getitem__
    # __loc = module[1000]
    # # import ipdb; ipdb.set_trace()
    # print(__loc.data, __loc.idx, __loc.ts)
    # __thd = module.write_items(1, 1, save_mode='tmp.txt')
    # __thd.join()

    # iterate_item
    module = LocationDecoder(
        datapath=LOC_PATH, sensor_name='local', source_sensor='car_center',
        cal_motion=True, intrpl_degree=3, cal_rotmat=True)

    for __loc in module.items(): # (sample_step=1000):
        ego2world = LocationDecoder.convert_coordinate(__loc, mode='ego2wld')
        print(__loc.data, __loc.idx, __loc.ts)

    # retrieve_item
    print(__loc.idx)
    __locs = module.retrieve_item(__loc.ts + 1)
    print(__locs)
