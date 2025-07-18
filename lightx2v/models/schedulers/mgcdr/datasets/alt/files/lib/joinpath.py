import os


def path_join(path: str, *other_paths: str) -> str:
    return str(os.path.join(path, *other_paths))


def uri_join(path: str, *other_paths: str) -> str:
    if len(other_paths) == 0:
        return path

    first_path = path
    if first_path.endswith('/'):
        first_path = first_path[:-1]

    last_path = other_paths[-1]
    if last_path.startswith('/'):
        last_path = last_path[1:]

    middle_paths = []
    for other_path in other_paths[:-1]:
        if other_path.startswith('/'):
            other_path = other_path[1:]
        if other_path.endswith('/'):
            other_path = other_path[:-1]
        middle_paths.append(other_path)

    return '/'.join([first_path, *middle_paths, last_path])

    # Imp. 2
    # other_paths = (other_path.lstrip('/') for other_path in other_paths)
    # return str(os.path.join(path, *other_paths))

    # Imp. 3
    # return '/'.join((path, *other_paths))
