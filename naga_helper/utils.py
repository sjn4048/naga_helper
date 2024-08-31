def get_leaf_path_by_name(data, key: str, value = None) -> list[str]:
    paths = []

    def _get_leaf_path_by_name(_data, _key: str, path: list[str]):
        if isinstance(_data, dict):
            if _key in _data and (not value or value == _data[_key]):
                path.append(_key)
                paths.append(".".join(path))
            for k, v in _data.items():
                new_path = path.copy()
                new_path.append(k)
                _get_leaf_path_by_name(v, _key, new_path)
        elif isinstance(_data, list):
            for idx, item in enumerate(_data):
                new_path = path.copy()
                new_path.append(str(idx))
                _get_leaf_path_by_name(item, _key, new_path)

    _get_leaf_path_by_name(data, key, [])
    return paths


class DotDict(dict):
    """dot.notation access to dictionary attributes recursively"""
    def __getattr__(self, k):
        ret = self.get(k)
        if isinstance(ret, dict):
            return DotDict(ret)
        return ret
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__