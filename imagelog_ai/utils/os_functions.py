import shutil
import os


def sort_listdir(directory_path: str) -> list:
    """Returns a sorted list of the files in the given directory path.

    Parameters
    ----------
    directory_path: str
            Path to the directory.
    """
    return sorted(os.listdir(directory_path))


def get_files_paths(src_path: str) -> list:
    """Returns an ordered list of the path of all files within the 
       subdirectory tree structure of the source directory.

    Parameters
    ----------
    src_path: str
            Path to the source directory.
    """
    list_file_paths = []

    for path, subdirs, files in os.walk(src_path):
        for name in files:
            list_file_paths.append(os.path.join(path, name))

    return sorted(list_file_paths)


def copytree(src_path: str, dst_path: str, dirs_exist_ok: bool = False) -> None:
    """Copies a directory tree structure ignoring files from source directories.

    Parameters
    ----------
    src_path: str
            Path to the source directory.

    dst_path: str
            Path where the directory tree structure will be copied.

    dirs_exist_ok: bool
            If dirs_exist_ok is false (the default) and dst already exists,
            a FileExistsError is raised. If dirs_exist_ok is true,
            the copying operation will continue if it encounters
            existing directories, and files within the dst tree
            will be overwritten by corresponding files from the src tree.
    """
    src_dir = src_path.split("/")[-1]

    def ignore_files(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    shutil.copytree(
        src=src_path,
        dst=os.path.join(dst_path, src_dir),
        dirs_exist_ok=dirs_exist_ok,
        ignore=ignore_files,
    )

def instantiate_obj_from_str(cls_name: str, cls_kwargs: dict, module: object):
    """Returns an object instantiated from its class name (string), 
       belonging to a given module.

    Parameters
    ----------
    cls_name: str
            Name of the class.

    cls_kwargs: dict
            Arguments for instantiating the object.

    module: object
            Module where the class is located.
    """    

    return getattr(module, cls_name)(**cls_kwargs)
