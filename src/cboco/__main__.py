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
    """Command to run. `stats` collects and displays statistics about the specified dataset. `intersect` takes two or more datasets, and outputs the common items between those datasets. `union` takes two or more datasets, and returns the combination of all of them. `subset` takes a dataset and returns a portion of the dataset, split by `split-method`. `unit` is the unit operation: reads a dataset and writes it out again - useful for catching or fixing bugs or formatting issues."""
    stats = 'stats'
    intersect = 'intersect'
    union = 'union'
    subset = 'subset'
    unit = 'unit'


class SplitMethod(enum.Enum):
    """Dataset split method."""
    random = 'random'


class CollisionStrategy(enum.Enum):
    """Dataset union collision handling strategy."""
    merge = 'merge'
    error = 'error'
    preserve = 'preserve'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=Command, action=EnumAction, help=Command.__doc__)
    parser.add_argument('file', type=str, nargs='+', help='Dataset json file paths')
    parser.add_argument('--output', '-o', type=str, required=False, help='Output file name. Only required for commands that would write out.')
    parser.add_argument('--split-method', type=SplitMethod, required=False, action=EnumAction, help=SplitMethod.__doc__)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--subset-size', type=int, default=0)
    parser.add_argument('--subset-by-total', action='store_true', default=False, help='Specify subset size by overall image, default is to striate subset by dir.')
    parser.add_argument('--union-collision-strategy', type=CollisionStrategy, action=EnumAction, help=CollisionStrategy.__doc__, default=CollisionStrategy.error)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == Command.stats:
        do_stats(args)
    elif args.command == Command.intersect:
        todo()
    elif args.command == Command.union:
        do_union(args)
    elif args.command == Command.subset:
        do_subset(args)
    elif args.command == Command.unit:
        do_unit(args)
    else:
        raise ValueError(f'Unhandled command {args.command}!')


def do_stats(args):
    for dsname in args.file:
        stats = Dataset.from_json(dsname).collect_statistics()
        completion_pc = stats.num_annotated_images * 100. / stats.num_images

        print(f'Dataset: {dsname}')
        print(f'Annotated images: {stats.num_annotated_images}/{stats.num_images} ({completion_pc:.1f}%)')
        print(f'Annotations by class:')
        for cls, n in stats.num_annotations_by_class.items():
            print(f' - {cls}: {n}')


def do_unit(args):
    assert len(args.file) == 1, 'unit accepts only 1 input dataset'
    assert args.output, 'unit requires `--output`'
    Dataset\
        .from_json(args.file[0])\
        .to_json(args.output)


def do_subset(args):
    assert len(args.file) == 1, 'subset accepts only 1 input dataset'
    assert args.output, 'subset requires `--output`'
    assert args.subset_size, 'subset requires `--subset-size`'
    assert args.split_method, 'subset requires `--split-method`'
    Dataset\
        .from_json(args.file[0])\
        .subset(method=args.split_method.value, by_dir=not args.subset_by_total, count=args.subset_size)\
        .to_json(args.output)


def do_union(args):
    assert len(args.file) > 1, 'unit requires at least 2 input datasets'
    assert args.output, 'union requires `--output`'
    datasets = [Dataset.from_json(fn) for fn in args.file]
    datasets[0]\
        .union(*datasets[1:], collision_strategy=args.union_collision_strategy.value)\
        .to_json(args.output)


def todo(_):
    raise NotImplementedError


if __name__ == '__main__':
    main()