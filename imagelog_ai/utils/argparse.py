from argparse import _SubParsersAction, ArgumentParser, Namespace
from types import TracebackType
from typing import Any, Callable, Iterable, Optional, Type


class ParserTree:
    """Command-line parsing

    Parameters:
    -----------
    parser: ArgumentParser
        Instance of ArgumentParser for parsing command line strings into Python objects.

    _subparsers: Optional[_SubParsersAction[ArgumentParser]]
        Defines the type of action to be taken when a subcommand is encountered at the command line. Defaults to None.

    Methods:
    --------
    call_endpoint
        Call the endpoint that triggers the desired route.
    """

    def __init__(
        self,
        name_or_prog: Optional[str] = None,
        *args: Any,
        endpoint: Optional[Callable[..., None]] = None,
        parent: Optional["ParserTree"] = None,
        **kwargs: Any,
    ) -> None:
        """Start the ParserTree class

        Parameters:
        -----------
        name_or_prog: string
            Name of file with flags who determine the route used to call the endpoint. Defaults to None.

        args: Any
            Non-keyworded arguments.

        endpoint: Callable[..., None]]
            Function of endpoint related to the name. Defaults to None.

        parent: 'ParserTree'
            Instance of parent ParserTree. Defaults to None.

        kwargs: Any
            Keyworded arguments.

        """
        if parent is None:
            parser = ArgumentParser(name_or_prog, *args, **kwargs)
        else:
            if parent._subparsers is None:
                parent._subparsers = parent.parser.add_subparsers()
            parser = parent._subparsers.add_parser(name_or_prog, *args, **kwargs)
        self.parser = parser
        if endpoint is None:

            def missing_positional_argument_error(**_: Any) -> None:
                print(
                    f"You must use some positional argument.\nRun 'python {self.parser.prog} -h' for help."
                )

            endpoint = missing_positional_argument_error
        self.parser.set_defaults(endpoint=endpoint)
        self._subparsers: Optional[_SubParsersAction[ArgumentParser]] = None

    def __enter__(self) -> "ParserTree":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        pass

    def call_endpoint(
        self,
        args: Optional[Iterable[str]] = None,
        namespace: Optional[Namespace] = None,
    ) -> None:
        """Start the ParserTree class

        Parameters:
        -----------
        args: Iterable[str]
            Non-keyworded arguments. Defaults to None.

        namespace: Namespace
            Instance of Namespace for storing attributes. Defaults to None.
        """
        kwargs = vars(self.parser.parse_args(args, namespace))
        endpoint = kwargs.pop("endpoint")
        # print(kwargs)
        endpoint(**kwargs)
