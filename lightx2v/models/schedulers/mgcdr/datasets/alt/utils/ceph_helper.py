import os
import configparser
import boto3
from tqdm import tqdm
import re

from .petrel_helper import petreloss_path

_S3_URI_PATTERN = re.compile(r'^(?:(?P<cluster>[^:]+):)?s3://(?P<bucket>[^/]+)/?(?P<key>(?:.+?)/?$)?', re.I)


def parse_uri(uri):
    m = _S3_URI_PATTERN.match(uri)
    if m:
        cluster, bucket, key = m.group('cluster'), m.group('bucket'), m.group('key')
        return cluster, bucket, key
    return None, None, None


class CephPetrel(object):
    def __init__(self) -> None:
        super().__init__()
        self.client = {}

        self.pre_init()

    def pre_init(self, conf_path=petreloss_path()):
        conf_path = os.path.expanduser(conf_path)
        if not os.path.exists(conf_path):
            return
        cfg = configparser.ConfigParser()
        cfg.read(conf_path)

        for sec in cfg.sections():
            try:

                ak = cfg.get(sec, 'access_key')
                sk = cfg.get(sec, 'secret_key')
                url = cfg.get(sec, 'host_base')

                client = boto3.client(service_name='s3', aws_access_key_id=ak, aws_secret_access_key=sk, endpoint_url=url, verify=False)
                self.client[sec] = client
            except Exception as e:
                pass

    def get_url(self, path, use_bee=True, expires_in=3600):
        cluster, bucket, key = parse_uri(path)
        if use_bee and cluster + '_bee' in self.client:
            cluster = cluster + '_bee'
        assert cluster in self.client, f'{cluster} not exist.'
        return self.client[cluster].generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expires_in,
        )


def get_bucket_size(s3, bucket):
    response = s3.head_bucket(Bucket=bucket)['ResponseMetadata']['HTTPHeaders']
    count = int(response['x-rgw-object-count'])
    size = int(response['x-rgw-bytes-used'])
    return {bucket: {'num': count, 'bytes': size, 'GB': size / 1024. / 1024. / 1024.}}


def list_bucket(client, bucket, kwargs):
    list_kwargs = {
        'Bucket': bucket,
        'PaginationConfig': {
            'PageSize': kwargs.get('page_size', 1000),
            'MaxItems': kwargs.get('max_size', None),
        }
    }
    if 'prefix' in kwargs and kwargs['prefix'] != '':
        list_kwargs['Prefix'] = kwargs['prefix']

    total = get_bucket_size(client, bucket)[bucket]['num']

    page_size = list_kwargs['PaginationConfig']['PageSize']

    page_number = (total - 1) // page_size + 1

    print('Bucket {bucket} totally contains {total} files; Iterating with PageSize={page_size}'.format(**locals()))

    paginator = client.get_paginator('list_objects')
    pages_generator = paginator.paginate(**list_kwargs)

    result = []
    for page in tqdm(pages_generator, total=page_number, desc='listing_pages'):
        contents = page.get('Contents', [])
        for content in contents:
            result.append(content['Key'])
    return result



def parallel_iterate_bucket_items(client, bucket):
    items = bucket.split('//')[-1].split('/', 1)
    if len(items) == 2:
        _bucket, _prefix = items
    elif len(items) == 1:
        _bucket = items[0]
        _prefix = ''
    else:
        raise AttributeError('{} not supported'.format(bucket))

    paginator = client.get_paginator('list_objects')

    prefix_results = paginator.paginate(Bucket=_bucket, Delimiter='/')

    prefixs = []
    for prefix in prefix_results.search('CommonPrefixes'):
        prefixs.append(prefix.get('Prefix'))

    # xx = list_ceph_folder(paginator, _bucket, prefixs[0])
    # import ipdb; ipdb.set_trace()

    commands = []
    print(prefixs)
    # import ipdb; ipdb.set_trace()
    for Prefix in prefixs:
        commands.append((list_ceph_folder, _bucket, Prefix))

    result_list, _ = ReturnMultiThreadHelper(commands, len(commands), prefix='finding status').run()

    out = []
    for item in result_list:
        out += item
    return out


def iterate_bucket_items(client, bucket):
    items = bucket.split('//')[-1].split('/', 1)
    if len(items) == 2:
        _bucket, _prefix = items
    elif len(items) == 1:
        _bucket = items[0]
        _prefix = ''
    else:
        raise AttributeError('{} not supported'.format(bucket))

    paginator = client.get_paginator('list_objects')
    page_iterator = paginator.paginate(Bucket=_bucket, Prefix=_prefix)

    for page in page_iterator:
        for item in page['Contents']:
            yield item


def get_bucket_items(client, bucket):
    result = []

    items = bucket.split('//')[-1].split('/', 1)
    if len(items) == 2:
        _, _prefix = items
    elif len(items) == 1:
        _ = items[0]
        _prefix = ''
    else:
        raise AttributeError('{} not supported'.format(bucket))

    for i in iterate_bucket_items(client=client, bucket=bucket):
        if _prefix == '':
            result.append(i['Key'])
        else:
            if _prefix.split('/')[0] == i['Key'].split('/')[0]:
                result.append(i['Key'].replace(_prefix + '/', ''))

    return result


def get_bucket_key(s):
    lst = s.split('/')
    if lst[0][-1] == ':':
        bucket = lst[2]
        key = '/'.join(lst[3:])
    else:
        bucket = ''
        key = '/'.join(lst)
    return bucket, key


def get_url(video, client=None):
    assert client is not None

    if 's3://' in video:
        bucket, key = get_bucket_key(video)
        url = client.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=864000000)
    else:
        url = video
    return url


def generate_presigned_url(metas, client=None):
    assert client is not None

    output = []
    for item in metas:
        item.url = get_url(item.path, client[item.petrel])
        output.append(item)

    return output
