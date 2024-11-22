import inspect
import errno
import os


def print_padded_string(
    string, message_length=100, padding=" ", border="#", border_size=3
):
    string_size = len(string)

    if string_size % 2 == 1:
        string += padding
        string_size += 1

    padding_size = message_length - string_size
    padding_size //= 2
    padding_size -= border_size

    print(
        f"{border_size * border}{padding_size * padding}{string}{padding_size * padding}{border_size * border}"
    )


def print_boxed_information(
    string_list, message_length=100, padding=" ", border="#", border_size=3
):
    print(message_length * border)
    for string in string_list:
        print_padded_string(string, message_length, padding, border, border_size)
    print(message_length * border)


def print_dictionary_information(
    information_dictionary: dict,
    value_box_length: int = 80,
    key_box_length: int = 40,
    title_string: str = None,
    border_char: str = "#",
    border_size: int = 3,
    pad_char: str = " ",
):
    message_length = key_box_length + value_box_length + (3 * border_size)
    border = border_size * border_char

    if title_string != None:
        print(message_length * border_char)
        print(
            border,
            title_string.center(message_length - (2 * border_size), pad_char),
            border,
            sep="",
        )

    print(message_length * border_char)

    for key, value in information_dictionary.items():
        if inspect.isclass(value):
            value_string = value.__name__
        elif type(value) == float:
            value_string = f"{value:.3E}"
        else:
            value_string = f"{value}"

        key_string = f"{key}"
        print(
            border,
            key_string.center(key_box_length, pad_char),
            border,
            value_string.center(value_box_length, pad_char),
            border,
            sep="",
        )

    print(message_length * border_char)


# def print_debug_number_open_files(letter):
#     ret = {}
#     base = "/proc/self/fd"
#     for num in os.listdir(base):
#         path = None
#         try:
#             path = os.readlink(os.path.join(base, num))
#         except OSError as err:
#             # Last FD is always the "listdir" one (which may be closed)
#             if err.errno != errno.ENOENT:
#                 raise
#         ret[int(num)] = path

#     print_boxed_information([f"{letter}", f"{len(ret):04d}"], message_length=24)


# def print_debug_list_open_files():
#     ret = {}
#     base = "/proc/self/fd"
#     for num in os.listdir(base):
#         path = None
#         try:
#             path = os.readlink(os.path.join(base, num))
#         except OSError as err:
#             # Last FD is always the "listdir" one (which may be closed)
#             if err.errno != errno.ENOENT:
#                 raise
#         ret[int(num)] = path
#     for k, v in ret.items():
#         print(k, v)

def print_debug_list_open_files(letter:str=None) -> None:
    base_dir = "/proc/self/fd" 
    returned_files = dict()
    for num in os.listdir(base_dir):
        path = None
        try:
            path = os.readlink(os.path.join(base_dir, num))
        except OSError as err:
            # Last FD is always the "listdir" one (which may be closed)
            if err.errno != errno.ENOENT:
                raise
        returned_files[int(num)] = path
    
    if isinstance(letter, str):
        print_boxed_information([f"{letter}", f"{len(returned_files):04d}"], message_length=24)
    else:
        for k, v in returned_files.items():
            print(k, v)
