import yaml
import re
import os.path
import json


class IncludeLoader(yaml.Loader):
    _cache = {}

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(IncludeLoader, self).__init__(stream)

        self.overwrite_keys = set(['overwrite@', 'override@'])

    def _split(self, path):
        splits = path.split('//', 1)
        if len(splits) == 1:
            return splits

        path, split = splits

        splits = re.split(r'//|\.', split)

        return [path] + splits

    # include
    def include(self, node):
        # splits = node.value.split('//')
        splits = self._split(node.value)
        node.value = splits[0]

        v = self.extract_file(self.construct_scalar(node))

        for k in splits[1:]:
            if isinstance(v, list):
                v = v[int(k)]
            else:
                v = v[k]
        return v

    # include, overwrite api
    def extract_file(self, filename):
        path = os.path.join(self._root, filename)

        if path in IncludeLoader._cache:
            return IncludeLoader._cache[path]

        with open(path, 'r') as f:
            v = yaml.load(f, IncludeLoader)

        IncludeLoader._cache[path] = v
        return v

    # overwrite
    def _process_overwrite(self, data, k):
        def key2tuple(key):
            current = data
            lst = []
            for k in key.split('.'):
                if isinstance(current, list):
                    k = int(k)
                try:
                    current = current[k]
                except:  # noqa
                    current = {}
                lst.append(k)
            return tuple(lst)

        def replace(overwrite_key, v):
            current = data
            keys = overwrite_key.split('.')

            # non-recursive
            for idx, k in enumerate(keys):
                if isinstance(current, dict):
                    if idx == len(keys) - 1:
                        current[k] = v
                        return

                    if k in current:
                        current = current[k]
                    else:
                        current[k] = {}
                        current = current[k]
                elif isinstance(current, list):
                    k = int(k)
                    assert k <= len(current)
                    if idx != len(keys) - 1:
                        if k < len(current):
                            current = current[k]
                        else:
                            current.append({})
                            current = current[k]
                    else:
                        if k < len(current):
                            current[k] = v
                        else:
                            current.append(v)
                else:
                    raise Exception(f'invalid replacement "{overwrite_key}" to {v}')

        overwrites = data[k]
        for overwrite_key, v in sorted(overwrites.items(), key=lambda x: key2tuple(x[0])):
            replace(overwrite_key, v)

        del data[k]
        return data

    # for overwrite
    def construct_document(self, node):
        data = super(IncludeLoader, self).construct_document(node)
        for k in self.overwrite_keys:
            if k in data:
                self._process_overwrite(data, k)
        return data


IncludeLoader.add_constructor('!include', IncludeLoader.include)


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fin:
        a = yaml.load(fin, IncludeLoader)

    print(json.dumps(a, indent=4))
