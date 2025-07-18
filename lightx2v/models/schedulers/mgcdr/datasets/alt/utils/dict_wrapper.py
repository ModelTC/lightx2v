class DictWrapper(object):
    def __init__(self, data):
        self.data = data

    def find(self, key):
        multi_keys = key.strip().split(".")

        data = self.data
        for single_key in multi_keys:
            if isinstance(data, dict):
                if single_key.isdigit():
                    if single_key in data:
                        data = data[single_key]
                    elif int(single_key) in data:
                        data = data[int(single_key)]
                    else:
                        raise NotImplementedError(single_key)
                else:
                    assert single_key in data, single_key
                    data = data[single_key]
            elif isinstance(data, list):
                assert single_key.isdigit(), single_key
                data = data[int(single_key)]
            else:
                raise NotImplementedError(single_key)

        return data
