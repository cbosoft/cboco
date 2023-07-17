import argparse
import enum
import os
from collections import defaultdict

from . import Dataset


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums

    https://stackoverflow.com/a/60750535
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


class Command(enum.Enum):
    stats = 'stats'
    intersect = 'intersect'
    union = 'union'
    subset = 'subset'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=Command, action=EnumAction)
    parser.add_argument('file', type=str, nargs='+')
    return parser.parse_args()


def do_stats(args):
    for dsname in args.file:
        stats = Dataset.from_json(dsname).collect_statistics()
        completion_pc = stats.num_annotated_images * 100. / stats.num_images

        print(f'Dataset: {dsname}')
        print(f'Annotated images: {stats.num_annotated_images}/{stats.num_images} ({completion_pc:.1f}%)')
        print(f'Annotations by class:')
        for cls, n in stats.num_annotations_by_class.items():
            print(f' - {cls}: {n}')


def todo(_):
    raise NotImplementedError


def main():
    args = parse_args()

    if args.command == Command.stats:
        do_stats(args)
    elif args.command == Command.intersect:
        todo()
    elif args.command == Command.union:
        todo()
    elif args.command == Command.subset:
        todo()
    else:
        raise ValueError(f'Unhandled command {args.command}!')


if __name__ == '__main__':
    main()