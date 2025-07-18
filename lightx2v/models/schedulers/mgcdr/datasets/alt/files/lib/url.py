def get_url_scheme(url: str):
    if '://' in url:
        return url.split('://', 1)[0]
    return ''
