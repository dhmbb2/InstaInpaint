import os
import shutil

from .dist_helper import get_rank


IS_MAST_ENV = False


class pathmgr:
    @staticmethod
    def ls(path):
        return os.listdir(path)

    @staticmethod
    def rm(path):
        return os.remove(path)

    @staticmethod
    def isfile(path):
        return os.path.isfile(path)

    @staticmethod
    def isdir(path):
        return os.path.isdir(path)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def copy_from_local(src, tgt, overwrite=True):
        if not overwrite and os.path.exists(tgt):
            return tgt
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        return shutil.copy2(src, tgt)

    @staticmethod
    def copy(src, tgt, overwrite=True):
        return pathmgr.copy_from_local(src, tgt, overwrite=overwrite)

    @staticmethod
    def open(*args, **kwargs):
        return open(*args, **kwargs)

    @staticmethod
    def mkdirs(path):
        return os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_local_path(path):
        return path


def may_download_to_local(path, subfolder=None):
    return os.path.join(path, subfolder) if subfolder else path


def replace_memcache_manifold(path):
    return path


def mkdirs(dirpath):
    if get_rank() == 0 and not pathmgr.isdir(dirpath):
        pathmgr.mkdirs(dirpath)
    return dirpath
