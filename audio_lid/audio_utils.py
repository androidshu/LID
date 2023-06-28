import hashlib


def generate_md5(raw_str):
    md5 = hashlib.md5()
    md5.update(raw_str.encode('utf-8'))
    md5_name = md5.hexdigest()
    return md5_name
