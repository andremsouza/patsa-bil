from typing import List, Tuple

import numpy as np
import torch


def _check_error_list(
    check_list: List[Tuple[bool, str]], variable_name: str, type_name: str
) -> None:
    """Organize and show the error messages when more than one error is raised

    Parameters
    ----------
    check_list : list
                    List of errors occurred.
    variable_name : string
                    Name of the variable where the error ocurred.
    type_name : string
                    Name of variable type.

    Raises
    ------
    TypeError
            TypeError raised with all the error messages.
    """
    failed_checks = np.array([i[0] for i in check_list])
    failed_checks = np.logical_not(failed_checks)

    messages = np.array([i[1] for i in check_list])

    if any(failed_checks):
        print("")
        error_message = f"`{variable_name}` is not valid `{type_name}`:"
        for error in messages[failed_checks]:
            error_message += f"\n\t{error}"
        print("")

        raise TypeError(error_message)


def _check_tensor_ndim(
    tensor: torch.Tensor, number_of_dimensions: int
) -> tuple[bool, str]:
    """Check if a tensor has the expected number of dimensions

    Parameters
    ----------
    tensor : torch.Tensor
            Tensor whose dimensions will be checked.
    number_of_dimensions : _type_
            The expected number of dimensions.

    Returns
    -------
    tuple[bool, str]
            A tuple containing a boolean indicating 'True' if the number of tensor dimensions is equal to expected, and the error message.
    """
    message = (
        f"Expected {number_of_dimensions} dimensions, got {tensor.ndim} dimensions"
    )
    check = tensor.ndim == number_of_dimensions
    return (check, message)


def _check_tensor_range(
    tensor: torch.Tensor, value_range: Tuple[float, float]
) -> tuple[bool, str]:
    """Check if the tensor values are in the expected range

    Parameters
    ----------
    tensor : torch.Tensor
            Tensor whose dimensions will be checked.
    values_range : tuple
            The expected range of values.

    Returns
    -------
    tuple[bool, str]
            A tuple containing a boolean indicating 'True' if the value range is within the expected, and the error message.
    """
    max = torch.max(tensor).item()
    min = torch.min(tensor).item()
    message = f"Expected values to be in {value_range} range, got ({min}, {max}) range"
    check = (value_range[0] <= min) and (max <= value_range[1])

    return (check, message)


def _check_tensor_positive(tensor: torch.Tensor) -> tuple[bool, str]:
    """Check if the tensor values are positive

    Parameters
    ----------
    tensor : torch.Tensor
            Tensor whose dimensions will be checked.

    Returns
    -------
    tuple[bool, str]
            A tuple containing a boolean indicating 'True' if the value range is within the expected, and the error message.
    """
    message = f"Expected tensor values to be positive."
    min = torch.min(tensor).item()
    check = min >= 0

    return (check, message)


def _check_tensor_dtype(tensor: torch.Tensor, torch_dtype) -> tuple[bool, str]:
    """Check if a tensor matches the expected type

    Parameters
    ----------
    tensor : torch.Tensor
            Tensor whose types will be checked.
    torch_dtype : _type_
            Desired type for the Tensor.

    Returns
    -------
    tuple[bool, str]
            A tuple containing a boolean indicating 'True' if the tensor type is the same as expected, and the error message.
    """
    message = f"Expected tensor value type to be {torch_dtype} , got {tensor.dtype}"
    check = tensor.dtype == torch_dtype
    return (check, message)


def _check_tensor_dimension_size(
    tensor: torch.Tensor, dimension, size
) -> tuple[bool, str]:
    """Check the size of a dimension

    Parameters
    ----------
    tensor : torch.Tensor
            Tensor whose dimension size will be checked.
    dimension : _type_
            Dimension position by index.
    size : _type_
            Desired dimension size.

    Returns
    -------
    tuple[bool, str]
            A tuple containing a boolean indicating 'True' if the tensor dimension size is equal to expected, and the error message.
    """
    message = f"Expected dimension {dimension} to have size {size}, dimension {dimension} has size {tensor.size(dim=dimension)}"
    check = tensor.size(dim=dimension) == size
    return (check, message)


def is_images_RGB_tensor(*, is_optional: bool = False, **kwargs: torch.Tensor) -> None:
    """Check if the input is a tensor of RGB images

    Parameters
    ----------
    is_optional : bool, optional
            Is `True` if the variable is optional, by default False.

    Raises
    ------
    TypeError
            If the variable passed is not a Tensor, a TypeError will be raised.
    """
    for name, arg in kwargs.items():
        if is_optional and arg is None:
            continue
        if not isinstance(arg, torch.Tensor):
            raise TypeError(f"`{name}` is {type(arg)}, expected `torch.Tensor`")

        check_list = []
        check_list.append(_check_tensor_dtype(arg, torch.float32))
        check_list.append(_check_tensor_ndim(arg, 4))
        if check_list[-1][0]:
            check_list.append(_check_tensor_dimension_size(arg, 1, 3))

        _check_error_list(check_list, name, "tensor of RGB images")
