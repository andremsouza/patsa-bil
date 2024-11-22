from typing import Any, Sequence, Tuple, Type, TypeVar, Union, List
from numbers import Number

import os


T_co = TypeVar("T_co", covariant=True)


_ClassOrTuple = Union[Type[T_co], Tuple[Type[T_co], ...]]


def _class_names(class_or_tuple: _ClassOrTuple) -> str:
    """Parser for classes names in the error messages

    Parameters
    ----------
    class_or_tuple : _ClassOrTuple
        Instance of a class or a tuple of instances to parse.

    Returns
    -------
    str
        Returns all the instances in string names.
    """
    if isinstance(class_or_tuple, Tuple):
        names = list(map(lambda arg: arg.__name__, class_or_tuple))
        return f'`{"`, `".join(names[:-1])}` or `{names[-1]}`'
    else:
        return f"`{class_or_tuple.__name__}`"


def dir_exists(
    dir_path: str,
    override: bool,
    **kwargs: Any,
) -> bool:
    """Checks if a directory exists.

    Parameters
    ----------
    dir_path : str
        Directory path.
    """

    if os.path.exists(dir_path):
        if override:
            print(f"\n\t{dir_path} already exists, and it will be overwritten.")
            return True
        else:
            raise ValueError(
                f"\n\t{dir_path} already exists.\n\tPlease create a diferent project for a new run."
            )


def experiment_was_run(
    project_name: str,
    experiment_name: str,
    result_path: str,
    override_experiment: bool,
    **kwargs: Any,
) -> None:
    """Checks if an experiment has already been run.

    Parameters
    ----------
    project_name : str
        Name of the project.
    experiment_name : str
        Name of the experiment.
    result_path : str
        Path of file that indicates the experiment has been run before.
        This path starts from the `projects/{project_name}/results/{experiment_name}/` folder

    Raises
    ------
    TypeError
        If the variable passed is not the expected instance type, a TypeError will be raised.
    ValueError
        If the number is not positive, a ValueError will be raised.
    """
    if override_experiment:
        return

    file_path = f"projects/{project_name}/results/{experiment_name}/{result_path}"

    if os.path.exists(file_path):
        raise ValueError(
            f"\n\t{experiment_name} experiment has already been executed.\n\tPlease create a diferent experiment for a new run."
        )


def is_positive(
    class_or_tuple: _ClassOrTuple[Number], *, optional: bool = False, **kwargs: Number
) -> None:
    """Check if the variable passed is a positive number

    Parameters
    ----------
    class_or_tuple : _ClassOrTuple[Number]
        Instance of expected type
    optional : bool, optional
        Is `True` if the variable is optional, by default False

    Raises
    ------
    TypeError
        If the variable passed is not the expected instance type, a TypeError will be raised.
    ValueError
        If the number is not positive, a ValueError will be raised.
    """
    message_tail = f"`must be a positive {_class_names(class_or_tuple)} value"
    for name, arg in kwargs.items():
        if optional and arg is None:
            continue
        if not isinstance(arg, class_or_tuple):
            raise TypeError(
                f"`{name}` is {type(arg)}, expected {_class_names(class_or_tuple)}"
            )

        if not arg > 0:
            raise ValueError(f"`{name}` {message_tail}")


def is_subclass(
    class_or_tuple: _ClassOrTuple, *, optional: bool = False, **kwargs: Type
) -> None:
    """Check if a class is a subclass of another

    Parameters
    ----------
    class_or_tuple : _ClassOrTuple
        Instance of the class to be checked
    optional : bool, optional
        Is `True` if the variable is optional, by default False

    Raises
    ------
    TypeError
        If the variable passed is not of the expected instance type, a TypeError will be raised.
    """
    message_tail = f"must be a subclass of {_class_names(class_or_tuple)}"
    for name, arg in kwargs.items():
        if optional and arg is None:
            continue

        if not issubclass(arg, class_or_tuple):
            raise TypeError(f"`{name}` {message_tail}")


def is_instance(
    class_or_tuple: _ClassOrTuple, *, optional: bool = False, **kwargs: Any
) -> None:
    """Check if the variable is an instance of a type

    Parameters
    ----------
    class_or_tuple : _ClassOrTuple
        Instance of the expected type.
    optional : bool, optional
        Is `True` if the variable is optional, by default False

    Raises
    ------
    TypeError
        If the variable passed is not of the expected instance type, a TypeError will be raised.
    """
    message_tail = f"must be instance of {_class_names(class_or_tuple)}"
    for name, arg in kwargs.items():
        if optional and arg is None:
            continue
        if not isinstance(arg, class_or_tuple):
            raise TypeError(f"`{name}` {message_tail}")


def is_instance_or_sequence(
    class_or_tuple: _ClassOrTuple, *, optional: bool = False, **kwargs: Any
) -> None:
    """Check if the argument passed is an instance or a list of instances of a type

    Parameters
    ----------
    class_or_tuple : _ClassOrTuple
        Instance of the class or type to be checked.
    optional : bool, optional
        Is `True` if the variable is optional, by default False

    Raises
    ------
    TypeError
        If the argument passed is None and the argument is not optional, a TypeError is raised.
    TypeError
        If the argument passed is not an instance or a list of instances of a type, a TypeError is raised.
    """
    message_tail = f"must be one instance or a sequence of instances of {_class_names(class_or_tuple)}"
    for name, arg in kwargs.items():
        if arg is None:
            if optional:
                continue
            else:
                raise TypeError(f"`{name}` is `None`")

        if not (
            isinstance(arg, class_or_tuple)
            or (
                isinstance(arg, Sequence)
                and all(map(lambda item: isinstance(item, class_or_tuple), arg))
            )
        ):
            raise TypeError(f"`{name}` {message_tail}")


def is_sequence(
    class_or_tuple: _ClassOrTuple,
    *,
    sequence_length: int = None,
    optional: bool = False,
    **kwargs: Any,
) -> None:
    """Check if the argument passed is an instance or a list of instances of a type

    Parameters
    ----------
    class_or_tuple : _ClassOrTuple
        Instance of the class or type to be checked.
    optional : bool, optional
        Is `True` if the variable is optional, by default False

    Raises
    ------
    TypeError
        If the argument passed is None and the argument is not optional, a TypeError is raised.
    TypeError
        If the argument passed is not an instance or a list of instances of a type, a TypeError is raised.
    """
    message_tail = f"must be a sequence of instances of {_class_names(class_or_tuple)}"
    for name, arg in kwargs.items():
        if arg is None:
            if optional:
                continue
            else:
                raise TypeError(f"`{name}` is `None`")

        if not (
            isinstance(arg, Sequence)
            and all(map(lambda item: isinstance(item, class_or_tuple), arg))
        ):
            raise TypeError(f"`{name}` {message_tail}")

        if sequence_length != None and len(arg) != sequence_length:
            raise TypeError(
                f"`{name}` is of length {len(arg)}, expected sequence of length {sequence_length}"
            )


def same_extension(list_of_paths: List[str]) -> Any:
    """Check if a list of files have the same extension.

    Parameters
    ----------
    list_of_paths : List[str]
        List of file paths.
    """

    extensions = list(set([path.split(".")[-1] for path in list_of_paths]))

    if len(extensions) > 1:
        raise TypeError(
            f"Files do not have the same extension. Extensions: [{extensions}]"
        )
    else:
        return extensions[0]
