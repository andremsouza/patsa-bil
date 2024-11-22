import random
from typing import List
import pkgutil
import importlib
import inspect


def list_filtering(input_list: List, filter_list: List) -> List:
    """Returns a copy of the input list with all elements present in the filter list removed.

    Parameters
    ----------
    input_list: List
            List to be filtered.
    filter_list: List
            List containing the elements to be filtered out.
    """
    return [x for x in input_list if x not in filter_list]


def remove_from_list(src_list: list, filter_list: list):
    return sorted(list(filter(lambda x: x not in filter_list, src_list)))


def k_fold_split(src_list: list, n_folds: int):
    return [src_list[i::n_folds] for i in range(n_folds)]


def flatten_list(src_list_of_lists):
    if all(isinstance(x, list) for x in src_list_of_lists):
        flat_list = []
        for row in src_list_of_lists:
            flat_list += row
        return flat_list
    else:
        return src_list_of_lists


def shuffle_two_lists(a: list, b: list):
    assert len(a) == len(b)
    start_state = random.getstate()
    random.shuffle(a)
    random.setstate(start_state)
    random.shuffle(b)


def list_non_abstract_classes(module_name, base_class):
    non_abstract_classes = {}

    module = importlib.import_module(module_name)
    module_path = module.__path__

    for finder, name, ispkg in pkgutil.walk_packages(module_path):
        full_module_name = f"{module_name}.{name}"
        try:
            sub_module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(sub_module, inspect.isclass):
                # Ensure the class is defined in the sub-module and is not abstract
                if (
                    obj.__module__ == full_module_name
                    and not inspect.isabstract(obj)
                    and (issubclass(obj, base_class))
                ):
                    non_abstract_classes[name] = obj
        except Exception as e:
            print(f"Error processing module {full_module_name}: {e}")

    return non_abstract_classes
