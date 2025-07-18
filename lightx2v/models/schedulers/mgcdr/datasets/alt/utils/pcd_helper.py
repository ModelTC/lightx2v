# flake8: noqa
# 刘凯给的点云相关Python代码，目前主要用一个从ceph读取的流转成点云功能

"""
Read and write PCL .pcd files in python.
dimatura@cmu.edu, 2013-2018

- TODO better API for wacky operations.
- TODO add a cli for common operations.
- TODO deal properly with padding
- TODO deal properly with multicount fields
- TODO better support for rgb nonsense
"""

import re
import struct
import copy
#from six import StringIO as sio
import io
import numpy as np
import warnings
import lzf

HAS_SENSOR_MSGS = True
try:
    from sensor_msgs.msg import PointField
    import numpy_pc2  # needs sensor_msgs
except ImportError:
    HAS_SENSOR_MSGS = False

__all__ = ['PointCloud',
           'point_cloud_to_path',
           'point_cloud_to_buffer',
           'point_cloud_to_fileobj',
           'point_cloud_from_path',
           'point_cloud_from_buffer',
           'point_cloud_from_fileobj',
           'make_xyz_point_cloud',
           'make_xyz_rgb_point_cloud',
           'make_xyz_label_point_cloud',
           'save_txt',
           'cat_point_clouds',
           'add_fields',
           'update_field',
           'build_ascii_fmtstr',
           'encode_rgb_for_pcl',
           'decode_rgb_from_pcl',
           'save_point_cloud',
           'save_point_cloud_bin',
           'save_point_cloud_bin_compressed',
           'pcd_type_to_numpy_type',
           'numpy_type_to_pcd_type',
           ]

if HAS_SENSOR_MSGS:
    pc2_pcd_type_mappings = [(PointField.INT8, ('I', 1)),
                             (PointField.UINT8, ('U', 1)),
                             (PointField.INT16, ('I', 2)),
                             (PointField.UINT16, ('U', 2)),
                             (PointField.INT32, ('I', 4)),
                             (PointField.UINT32, ('U', 4)),
                             (PointField.FLOAT32, ('F', 4)),
                             (PointField.FLOAT64, ('F', 8))]
    pc2_type_to_pcd_type = dict(pc2_pcd_type_mappings)
    pcd_type_to_pc2_type = dict((q, p) for (p, q) in pc2_pcd_type_mappings)
    __all__.extend(['pcd_type_to_pc2_type', 'pc2_type_to_pcd_type'])

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    """ Parse header of PCD files.
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def write_header(metadata, rename_padding=False):
    """ Given metadata as dictionary, return a string header.
    """
    template = """\
VERSION {version}
FIELDS {fields}
SIZE {size}
TYPE {type}
COUNT {count}
WIDTH {width}
HEIGHT {height}
VIEWPOINT {viewpoint}
POINTS {points}
DATA {data}
"""
    str_metadata = metadata.copy()

    if not rename_padding:
        str_metadata['fields'] = ' '.join(metadata['fields'])
    else:
        new_fields = []
        for f in metadata['fields']:
            if f == '_':
                new_fields.append('padding')
            else:
                new_fields.append(f)
        str_metadata['fields'] = ' '.join(new_fields)
    str_metadata['size'] = ' '.join(map(str, metadata['size']))
    str_metadata['type'] = ' '.join(metadata['type'])
    str_metadata['count'] = ' '.join(map(str, metadata['count']))
    str_metadata['width'] = str(metadata['width'])
    str_metadata['height'] = str(metadata['height'])
    str_metadata['viewpoint'] = ' '.join(map(str, metadata['viewpoint']))
    str_metadata['points'] = str(metadata['points'])
    tmpl = template.format(**str_metadata)
    return tmpl


def _metadata_is_consistent(metadata):
    """ Sanity check for metadata. Just some basic checks.
    """
    checks = []
    required = ('version', 'fields', 'size', 'width', 'height', 'points',
                'viewpoint', 'data')
    for f in required:
        if f not in metadata:
            print('%s required' % f)
    checks.append((lambda m: all([k in m for k in required]),
                   'missing field'))
    checks.append((lambda m: len(list(m['type'])) == len(list(m['count'])) ==
                   len(list(m['fields'])),
                   'length of type, count and fields must be equal'))
    checks.append((lambda m: m['height'] > 0,
                   'height must be greater than 0'))
    checks.append((lambda m: m['width'] > 0,
                   'width must be greater than 0'))
    checks.append((lambda m: m['points'] > 0,
                   'points must be greater than 0'))
    checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                   'binary_compressed'),
                   'unknown data type:'
                   'should be ascii/binary/binary_compressed'))
    ok = True
    for check, msg in checks:
        if not check(metadata):
            print('error:', msg)
            ok = False
    return ok

# def pcd_type_to_numpy(pcd_type, pcd_sz):
#     """ convert from a pcd type string and size to numpy dtype."""
#     typedict = {'F' : { 4:np.float32, 8:np.float64 },
#                 'I' : { 1:np.int8, 2:np.int16, 4:np.int32, 8:np.int64 },
#                 'U' : { 1:np.uint8, 2:np.uint16, 4:np.uint32 , 8:np.uint64 }}
#     return typedict[pcd_type][pcd_sz]


def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.

    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.

    *TODO* allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in xrange(c)])
            typenames.extend([np_type]*c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def build_ascii_fmtstr(pc):
    """ Make a format string for printing to ascii.

    Note %.8f is minimum for rgb.
    """
    fmtstr = []
    for t, cnt in zip(pc.type, pc.count):
        if t == 'F':
            fmtstr.extend(['%.10f']*cnt)
        elif t == 'I':
            fmtstr.extend(['%d']*cnt)
        elif t == 'U':
            fmtstr.extend(['%u']*cnt)
        else:
            raise ValueError("don't know about type %s" % t)
    return fmtstr


def parse_ascii_pc_data(f, dtype, metadata):
    """ Use numpy to parse ascii pointcloud data.
    """
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    rowstep = metadata['points']*dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    """ Parse lzf-compressed data.
    Format is undocumented but seems to be:
    - compressed size of data (uint32)
    - uncompressed size of data (uint32)
    - compressed data
    - junk
    """
    fmt = 'II'
    compressed_size, uncompressed_size =\
        struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    compressed_data = f.read(compressed_size)
    # TODO what to use as second argument? if buf is None
    # (compressed > uncompressed)
    # should we read buf as raw binary?
    buf = lzf.decompress(compressed_data, uncompressed_size)
    if len(buf) != uncompressed_size:
        raise IOError('Error decompressing data')
    # the data is stored field-by-field
    pc_data = np.zeros(metadata['width'], dtype=dtype)
    ix = 0
    for dti in range(len(dtype)):
        dt = dtype[dti]
        bytes = dt.itemsize * metadata['width']
        column = np.fromstring(buf[ix:(ix+bytes)], dt)
        pc_data[dtype.names[dti]] = column
        ix += bytes
    return pc_data


def point_cloud_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = (f.readline().strip()).decode("utf-8")
        header.append(ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            metadata_temp = copy.deepcopy(metadata)
            dtype = _build_dtype(metadata_temp)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"')
    #print(header)
    return PointCloud(metadata, pc_data)


def point_cloud_from_path(fname):
    """ load point cloud in binary format
    """
    with open(fname, 'rb') as f:
        pc = point_cloud_from_fileobj(f)
    return pc


def point_cloud_from_buffer(buf):
    #fileobj = sio.StringIO(buf)
    fileobj = io.BytesIO(buf)
    pc = point_cloud_from_fileobj(fileobj)
    fileobj.close()  # necessary?
    return pc


def point_cloud_to_fileobj(pc, fileobj, data_compression=None):
    """ Write pointcloud as .pcd to fileobj.
    If data_compression is not None it overrides pc.data.
    """
    metadata = pc.get_metadata()
    if data_compression is not None:
        data_compression = data_compression.lower()
        assert(data_compression in ('ascii', 'binary', 'binary_compressed'))
        metadata['data'] = data_compression

    header = write_header(metadata)
    fileobj.write(header)
    if metadata['data'].lower() == 'ascii':
        fmtstr = build_ascii_fmtstr(pc)
        np.savetxt(fileobj, pc.pc_data, fmt=fmtstr)
    elif metadata['data'].lower() == 'binary':
        fileobj.write(pc.pc_data.tostring('C'))
    elif metadata['data'].lower() == 'binary_compressed':
        # TODO
        # a '_' field is ignored by pcl and breakes compressed point clouds.
        # changing '_' to '_padding' or other name fixes this.
        # admittedly padding shouldn't be compressed in the first place.
        # reorder to column-by-column
        uncompressed_lst = []
        for fieldname in pc.pc_data.dtype.names:
            column = np.ascontiguousarray(pc.pc_data[fieldname]).tostring('C')
            uncompressed_lst.append(column)
        uncompressed = ''.join(uncompressed_lst)
        uncompressed_size = len(uncompressed)
        # print("uncompressed_size = %r"%(uncompressed_size))
        buf = lzf.compress(uncompressed)
        if buf is None:
            # compression didn't shrink the file
            # TODO what do to do in this case when reading?
            buf = uncompressed
            compressed_size = uncompressed_size
        else:
            compressed_size = len(buf)
        fmt = 'II'
        fileobj.write(struct.pack(fmt, compressed_size, uncompressed_size))
        fileobj.write(buf)
    else:
        raise ValueError('unknown DATA type')
    # we can't close because if it's stringio buf then we can't get value after


def point_cloud_to_path(pc, fname):
    with open(fname, 'w') as f:
        point_cloud_to_fileobj(pc, f)


def point_cloud_to_buffer(pc, data_compression=None):
    fileobj = sio.StringIO()
    point_cloud_to_fileobj(pc, fileobj, data_compression)
    return fileobj.getvalue()


def save_point_cloud(pc, fname):
    """ Save pointcloud to fname in ascii format.
    """
    with open(fname, 'w') as f:
        point_cloud_to_fileobj(pc, f, 'ascii')


def save_point_cloud_bin(pc, fname):
    """ Save pointcloud to fname in binary format.
    """
    with open(fname, 'w') as f:
        point_cloud_to_fileobj(pc, f, 'binary')


def save_point_cloud_bin_compressed(pc, fname):
    """ Save pointcloud to fname in binary compressed format.
    """
    with open(fname, 'w') as f:
        point_cloud_to_fileobj(pc, f, 'binary_compressed')


def save_xyz_label(pc, fname, use_default_lbl=False):
    """ Save a simple (x y z label) pointcloud, ignoring all other features.
    Label is initialized to 1000, for an obscure program I use.
    """
    md = pc.get_metadata()
    if not use_default_lbl and ('label' not in md['fields']):
        raise Exception('label is not a field in this point cloud')
    with open(fname, 'w') as f:
        for i in xrange(pc.points):
            x, y, z = ['%.4f' % d for d in (
                pc.pc_data['x'][i], pc.pc_data['y'][i], pc.pc_data['z'][i]
                )]
            lbl = '1000' if use_default_lbl else pc.pc_data['label'][i]
            f.write(' '.join((x, y, z, lbl))+'\n')


def save_xyz_intensity_label(pc, fname, use_default_lbl=False):
    """ Save XYZI point cloud.
    """
    md = pc.get_metadata()
    if not use_default_lbl and ('label' not in md['fields']):
        raise Exception('label is not a field in this point cloud')
    if 'intensity' not in md['fields']:
        raise Exception('intensity is not a field in this point cloud')
    with open(fname, 'w') as f:
        for i in xrange(pc.points):
            x, y, z = ['%.4f' % d for d in (
                pc.pc_data['x'][i], pc.pc_data['y'][i], pc.pc_data['z'][i]
                )]
            intensity = '%.4f' % pc.pc_data['intensity'][i]
            lbl = '1000' if use_default_lbl else pc.pc_data['label'][i]
            f.write(' '.join((x, y, z, intensity, lbl))+'\n')


def save_txt(pc, fname, header=True):
    """ Save to csv-style text file, separated by spaces.

    TODO:
    - support multi-count fields.
    - other delimiters.
    """
    with open(fname, 'w') as f:
        if header:
            header_lst = []
            for field_name, cnt in zip(pc.fields, pc.count):
                if cnt == 1:
                    header_lst.append(field_name)
                else:
                    for c in xrange(cnt):
                        header_lst.append('%s_%04d' % (field_name, c))
            f.write(' '.join(header_lst)+'\n')
        fmtstr = build_ascii_fmtstr(pc)
        np.savetxt(f, pc.pc_data, fmt=fmtstr)


def update_field(pc, field, pc_data):
    """ Updates field in-place.
    """
    pc.pc_data[field] = pc_data
    return pc


def add_fields(pc, metadata, pc_data):
    """ Builds copy of pointcloud with extra fields.

    Multi-count fields are sketchy, yet again.
    """
    if len(set(metadata['fields']).intersection(set(pc.fields))) > 0:
        raise Exception("Fields with that name exist.")

    if pc.points != len(pc_data):
        raise Exception("Mismatch in number of points.")

    new_metadata = pc.get_metadata()
    new_metadata['fields'].extend(metadata['fields'])
    new_metadata['count'].extend(metadata['count'])
    new_metadata['size'].extend(metadata['size'])
    new_metadata['type'].extend(metadata['type'])

    # parse metadata to add
    # TODO factor this
    fieldnames, typenames = [], []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in xrange(c)])
            typenames.extend([np_type]*c)
    dtype = zip(fieldnames, typenames)
    # new dtype. could be inferred?
    new_dtype = [(f, pc.pc_data.dtype[f])
                 for f in pc.pc_data.dtype.names] + dtype

    new_data = np.empty(len(pc.pc_data), new_dtype)
    for n in pc.pc_data.dtype.names:
        new_data[n] = pc.pc_data[n]
    for n, n_tmp in zip(fieldnames, pc_data.dtype.names):
        new_data[n] = pc_data[n_tmp]

    # TODO maybe just all the metadata in the dtype.
    # TODO maybe use composite structured arrays for fields with count > 1
    newpc = PointCloud(new_metadata, new_data)
    return newpc


def cat_point_clouds(pc1, pc2):
    """ Concatenate two point clouds into bigger point cloud.
    Point clouds must have same metadata.
    """
    if len(pc1.fields) != len(pc2.fields):
        raise ValueError("Pointclouds must have same fields")
    new_metadata = pc1.get_metadata()
    new_data = np.concatenate((pc1.pc_data, pc2.pc_data))
    # TODO this only makes sense for unstructured pc?
    new_metadata['width'] = pc1.width+pc2.width
    new_metadata['points'] = pc1.points+pc2.points
    pc3 = PointCloud(new_metadata, new_data)
    return pc3


def make_xyz_point_cloud(xyz, metadata=None):
    """ Make a pointcloud object from xyz array.
    xyz array is cast to float32.
    """
    md = {'version': .7,
          'fields': ['x', 'y', 'z'],
          'size': [4, 4, 4],
          'type': ['F', 'F', 'F'],
          'count': [1, 1, 1],
          'width': len(xyz),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyz),
          'data': 'binary'}
    if metadata is not None:
        md.update(metadata)
    xyz = xyz.astype(np.float32)
    pc_data = xyz.view(np.dtype([('x', np.float32),
                                 ('y', np.float32),
                                 ('z', np.float32)]))
    # pc_data = np.rec.fromarrays([xyz[:,0], xyz[:,1], xyz[:,2]], dtype=dt)
    # data = np.rec.fromarrays([xyz.T], dtype=dt)
    pc = PointCloud(md, pc_data)
    return pc


def make_xyz_rgb_point_cloud(xyz_rgb, metadata=None):
    """ Make a pointcloud object from xyz array.
    xyz array is assumed to be float32.
    rgb is assumed to be encoded as float32 according to pcl conventions.
    """
    md = {'version': .7,
          'fields': ['x', 'y', 'z', 'rgb'],
          'count': [1, 1, 1, 1],
          'width': len(xyz_rgb),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyz_rgb),
          'type': ['F', 'F', 'F', 'F'],
          'size': [4, 4, 4, 4],
          'data': 'binary'}
    if xyz_rgb.dtype != np.float32:
        raise ValueError('array must be float32')
    if metadata is not None:
        md.update(metadata)
    pc_data = xyz_rgb.view(np.dtype([('x', np.float32),
                                     ('y', np.float32),
                                     ('z', np.float32),
                                     ('rgb', np.float32)])).squeeze()
    # pc_data = np.rec.fromarrays([xyz[:,0], xyz[:,1], xyz[:,2]], dtype=dt)
    # data = np.rec.fromarrays([xyz.T], dtype=dt)
    pc = PointCloud(md, pc_data)
    return pc


def encode_rgb_for_pcl(rgb):
    """ Encode bit-packed RGB for use with PCL.

    :param rgb: Nx3 uint8 array with RGB values.
    :rtype: Nx1 float32 array with bit-packed RGB, for PCL.
    """
    assert(rgb.dtype == np.uint8)
    assert(rgb.ndim == 2)
    assert(rgb.shape[1] == 3)
    rgb = rgb.astype(np.uint32)
    rgb = np.array((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | (rgb[:, 2] << 0),
                   dtype=np.uint32)
    rgb.dtype = np.float32
    return rgb


def decode_rgb_from_pcl(rgb):
    """ Decode the bit-packed RGBs used by PCL.

    :param rgb: An Nx1 array.
    :rtype: Nx3 uint8 array with one column per color.
    """

    rgb = rgb.copy()
    rgb.dtype = np.uint32
    r = np.asarray((rgb >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb & 255, dtype=np.uint8)
    rgb_arr = np.zeros((len(rgb), 3), dtype=np.uint8)
    rgb_arr[:, 0] = r
    rgb_arr[:, 1] = g
    rgb_arr[:, 2] = b
    return rgb_arr


def make_xyz_label_point_cloud(xyzl, label_type='f'):
    """ Make XYZL point cloud from numpy array.

    TODO i labels?
    """
    md = {'version': .7,
          'fields': ['x', 'y', 'z', 'label'],
          'count': [1, 1, 1, 1],
          'width': len(xyzl),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyzl),
          'data': 'ASCII'}
    if label_type.lower() == 'f':
        md['size'] = [4, 4, 4, 4]
        md['type'] = ['F', 'F', 'F', 'F']
    elif label_type.lower() == 'u':
        md['size'] = [4, 4, 4, 1]
        md['type'] = ['F', 'F', 'F', 'U']
    else:
        raise ValueError('label type must be F or U')
    # TODO use .view()
    xyzl = xyzl.astype(np.float32)
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                   ('label', np.float32)])
    pc_data = np.rec.fromarrays([xyzl[:, 0], xyzl[:, 1], xyzl[:, 2],
                                 xyzl[:, 3]], dtype=dt)
    pc = PointCloud(md, pc_data)
    return pc


class PointCloud(object):
    """ Wrapper for point cloud data.

    The variable members of this class parallel the ones used by
    the PCD metadata (and similar to PCL and ROS PointCloud2 messages),

    ``pc_data`` holds the actual data as a structured numpy array.

    The other relevant metadata variables are:

    - ``version``: Version, usually .7
    - ``fields``: Field names, e.g. ``['x', 'y' 'z']``.
    - ``size.`: Field sizes in bytes, e.g. ``[4, 4, 4]``.
    - ``count``: Counts per field e.g. ``[1, 1, 1]``. NB: Multi-count field
      support is sketchy.
    - ``width``: Number of points, for unstructured point clouds (assumed by
      most operations).
    - ``height``: 1 for unstructured point clouds (again, what we assume most
      of the time.
    - ``viewpoint``: A pose for the viewpoint of the cloud, as
      x y z qw qx qy qz, e.g. ``[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]``.
    - ``points``: Number of points.
    - ``type``: Data type of each field, e.g. ``[F, F, F]``.
    - ``data``: Data storage format. One of ``ascii``, ``binary`` or ``binary_compressed``.

    See `PCL docs <http://pointclouds.org/documentation/tutorials/pcd_file_format.php>`__
    for more information.
    """

    def __init__(self, metadata, pc_data):
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        """ returns copy of metadata """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def check_sanity(self):
        # pdb.set_trace()
        md = self.get_metadata()
        #self.count = copy.deepcopy((md['count']))
        fields_temp = copy.deepcopy(self.fields)
        count_temp = copy.deepcopy(self.count)
        type_temp = copy.deepcopy(self.type)
        assert(_metadata_is_consistent(md))
        assert(len(self.pc_data) == self.points)
        assert(self.width*self.height == self.points)
        assert(len(list(fields_temp)) == len(list(count_temp)))
        assert(len(list(fields_temp)) == len(list(type_temp)))

    def save(self, fname):
        self.save_pcd(fname, 'ascii')

    def save_pcd(self, fname, compression=None, **kwargs):
        if 'data_compression' in kwargs:
            warnings.warn('data_compression keyword is deprecated for'
                          ' compression')
            compression = kwargs['data_compression']
        with open(fname, 'w') as f:
            point_cloud_to_fileobj(self, f, compression)

    def save_pcd_to_fileobj(self, fileobj, compression=None, **kwargs):
        if 'data_compression' in kwargs:
            warnings.warn('data_compression keyword is deprecated for'
                          ' compression')
            compression = kwargs['data_compression']
        point_cloud_to_fileobj(self, fileobj, compression)

    def save_pcd_to_buffer(self, compression=None, **kwargs):
        if 'data_compression' in kwargs:
            warnings.warn('data_compression keyword is deprecated for'
                          ' compression')
            compression = kwargs['data_compression']
        return point_cloud_to_buffer(self, compression)

    def save_txt(self, fname):
        save_txt(self, fname)

    def save_xyz_label(self, fname, **kwargs):
        save_xyz_label(self, fname, **kwargs)

    def save_xyz_intensity_label(self, fname, **kwargs):
        save_xyz_intensity_label(self, fname, **kwargs)

    def copy(self):
        new_pc_data = np.copy(self.pc_data)
        new_metadata = self.get_metadata()
        return PointCloud(new_metadata, new_pc_data)

    def to_msg(self):
        if not HAS_SENSOR_MSGS:
            raise Exception('ROS sensor_msgs not found')
        # TODO is there some metadata we want to attach?
        return numpy_pc2.array_to_pointcloud2(self.pc_data)

    @staticmethod
    def from_path(fname):
        return point_cloud_from_path(fname)

    @staticmethod
    def from_fileobj(fileobj):
        return point_cloud_from_fileobj(fileobj)

    @staticmethod
    def from_buffer(buf):
        return point_cloud_from_buffer(buf)

    @staticmethod
    def from_array(arr):
        """ create a PointCloud object from an array.
        """
        pc_data = arr.copy()
        md = {'version': .7,
              'fields': [],
              'size': [],
              'count': [],
              'width': 0,
              'height': 1,
              'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              'points': 0,
              'type': [],
              'data': 'binary_compressed'}
        md['fields'] = pc_data.dtype.names
        for field in md['fields']:
            type_, size_ =\
                numpy_type_to_pcd_type[pc_data.dtype.fields[field][0]]
            md['type'].append(type_)
            md['size'].append(size_)
            # TODO handle multicount
            md['count'].append(1)
        md['width'] = len(pc_data)
        md['points'] = len(pc_data)
        pc = PointCloud(md, pc_data)
        return pc

    @staticmethod
    def from_msg(msg, squeeze=True):
        """ from pointcloud2 msg
        squeeze: fix when clouds get 1 as first dim
        """
        if not HAS_SENSOR_MSGS:
            raise NotImplementedError('ROS sensor_msgs not found')
        md = {'version': .7,
              'fields': [],
              'size': [],
              'count': [],
              'width': msg.width,
              'height': msg.height,
              'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              'points': 0,
              'type': [],
              'data': 'binary_compressed'}
        for field in msg.fields:
            md['fields'].append(field.name)
            t, s = pc2_type_to_pcd_type[field.datatype]
            md['type'].append(t)
            md['size'].append(s)
            # TODO handle multicount correctly
            if field.count > 1:
                warnings.warn('fields with count > 1 are not well tested')
            md['count'].append(field.count)
        pc_array = numpy_pc2.pointcloud2_to_array(msg)
        pc_data = pc_array.reshape(-1)
        md['height'], md['width'] = pc_array.shape
        md['points'] = len(pc_data)
        pc = PointCloud(md, pc_data)
        return pc