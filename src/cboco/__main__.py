import argparse
import enum
from typing import List, Optional
import os
from collections import defaultdict

from . import Dataset
from . import evaluate_dataset


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


class SplitMethod(enum.Enum):
    """Dataset split method."""
    random = 'random'


class CollisionStrategy(enum.Enum):
    """Dataset union collision handling strategy."""
    merge = 'merge'
    error = 'error'
    preserve = 'preserve'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('python -m cboco')
    subps = parser.add_subparsers(dest='command')
    
    stats_command = subps.add_parser('stats', help='Get stats about a dataset')
    stats_command.add_argument('dataset', type=str, nargs='+', help='Dataset(s) to look at.')
    stats_command.add_argument('--scale', '-s', type=str, action='append', help='string defining pixel size in format "<filename regex>:<pixel size or ratio>"')
    stats_command.add_argument('--unit', type=str, default='μm', help='unit for scaled length. Default is micron.')

    subset_command = subps.add_parser('subset', help='Carve a portion off a dataset')
    subset_command.add_argument('--method', type=SplitMethod, default=SplitMethod.random, action=EnumAction, help=SplitMethod.__doc__)
    subset_command.add_argument('--size', type=int, default=100, help='Size of subset portion.')
    subset_command.add_argument('--by-total', action='store_true', default=False, help='Specify subset size by overall image, default is to stratify subset by dir.')

    union_command = subps.add_parser('union', help='Join two or more datasets together.')
    union_command.add_argument('--collision-strategy', type=CollisionStrategy, action=EnumAction, help=CollisionStrategy.__doc__, default=CollisionStrategy.error)
    union_command.add_argument('dataset1', type=str, nargs=1, help='First dataset to combine.')
    union_command.add_argument('datasets', type=str, nargs='+', help='Rest of the datasets to combine.')
    union_command.add_argument('--output', '-o', type=str, help='Name of resulting combined dataset.')

    # TODO
    # intersect_command = subps.add_parser('intersect', help='Get intersection of two or more datasets')

    unit_command = subps.add_parser('unit', help='Open dataset and do nothing. Useful for catching errors. Optionally write formatted file out.')
    unit_command.add_argument('dataset', type=str, nargs=1, help='Dataset to look at.')
    unit_command.add_argument('--output', '-o', type=str, required=False, help='Name of resulting combined dataset.')

    eval_command = subps.add_parser('eval', help='Evaluate one or more datasets with respect to a truth dataset.')
    eval_command.add_argument('truth', type=str, help='Dataset with "ground truth" annotations.')
    eval_command.add_argument('preds', type=str, nargs='+', help='Dataset(s) containing prediction object detections.')
    eval_command.add_argument('--output', '-o', type=str, required=False, help='Filename to write evaluation report to for each of dataset $preds.')
    eval_command.add_argument('--thresholds', '-t', type=str, default='coco', help='Comma-separated list of IoU thresholds (integers 0-100) to use to calculate metrics. Set to "coco" to use thresholds 50 to 95 in steps of 5.')
    eval_command.add_argument('--values', '-v', type=str, nargs=1, default='AP_50,mAP,mF1', help='Comma-separated list of metrics to display. Set to "all" to display all. Default only valid for multiple IoU thresholds.')
    eval_command.add_argument('--class-agnostic', action='store_true', help='Perform evaluation with no regard for particle class.')

    args = parser.parse_args()
    command = str(args.command)
    del args.command
    return command, args.__dict__


def main():
    command, kwargs = parse_args()

    if command == 'stats':
        do_stats(**kwargs)
    elif command == 'intersect':
        todo()
    elif command == 'union':
        do_union(**kwargs)
    elif command == 'subset':
        do_subset(**kwargs)
    elif command == 'unit':
        do_unit(**kwargs)
    elif command == 'eval':
        do_eval(**kwargs)
    else:
        raise ValueError(f'Unhandled command {command}!')


def do_stats(*, dataset: List[str], scale: List[str], unit: str):
    for dsname in dataset:
        stats = Dataset.from_json(dsname).collect_statistics(scale)
        completion_pc = stats.num_annotated_images * 100. / stats.num_images

        print(f'Dataset: {dsname}')
        print(f'Annotated images: {stats.num_annotated_images}/{stats.num_images} ({completion_pc:.1f}%)')
        print(f'Annotations by class:')
        for cls, n in stats.num_annotations_by_class.items():
            print(f' - {cls}: {n}')

        # only use unit if scale is valid
        unit = 'px' if not scale else unit
        print(f'Mean length {stats.mean_length:.1f} {unit} (σ={stats.stddev_length:.2f} {unit})')
        print(f'Mean width {stats.mean_width:.1f} {unit} (σ={stats.stddev_width:.2f} {unit})')
        print(f'Mean aspect_ratio {stats.mean_aspect_ratio:.3f} (σ={stats.stddev_aspect_ratio:.4f})')


def do_unit(*, dataset: str, output: Optional[str]):
    ds = Dataset.from_json(dataset)
    if output is not None:
        ds.to_json(output)


def do_subset(*, dataset: str, output: str, size: int, split_method: SplitMethod, by_total: bool):
    Dataset\
        .from_json(dataset)\
        .subset(method=split_method.value, by_dir=not by_total, count=size)\
        .copy_files(os.path.dirname(output))\
        .to_json(output)


def do_union(*, dataset1: str, dataset2: List[str], output: str, collision_strategy: CollisionStrategy):
    datasets = [Dataset.from_json(fn) for fn in [dataset1, *dataset2]]
    datasets[0]\
        .union(*datasets[1:], collision_strategy=collision_strategy.value)\
        .copy_files(os.path.dirname(output))\
        .to_json(output)


def do_eval(*, truth: str, preds: List[str], output: Optional[str], thresholds: str, values: str, class_agnostic: bool):
    if thresholds == 'coco':
        thresholds = [float(v)*0.01 for v in range(50, 100, 5)]
    else:
        thresholds = [float(v.strip())*0.01 for v in thresholds.split(',')]
    
    ds_truth = Dataset.from_json(truth)
    results_by_preds = {}
    for pred in preds:
        ds_preds = Dataset.from_json(pred)
        results_by_preds[pred] = evaluate_dataset(
            ds_preds, ds_truth,
            iou_thresh=thresholds,
            class_agnostic=class_agnostic,
        )
    
    possible_keys = set(list(results_by_preds.values())[0].keys())
    if values == 'all':
        keys = list(possible_keys)
    else:
        keys = [v.strip() for v in values.split(',')]
        assert possible_keys.issuperset(keys), \
            f'invalid values specified: "{set(keys).difference(possible_keys.union(keys))}" out of "{possible_keys}"'
    
    results = {k: [] for k in keys}
    for pred, pred_results in results_by_preds.items():
        for k in keys:
            results[k].append(pred_results[k])

    print('Evalation results')
    print(f'\nTruth: {truth}\n')
    print(' {:20} | {}'.format('Metrics \\ Preds', ' | '.join([f'{p[-20:]:20}' for p in preds])))
    for mname, mvalues in results.items():
        print(' {:20} | {}'.format(mname, ' | '.join([f'{mvalue:.4f}'.ljust(20) for mvalue in mvalues])))
    
    if output is not None:
        if not output.endswith('.txt'):
            output = output + '.txt'
        else:
            output = output
        print(f'Writing report to "{output}"')
        with open(output, 'w') as f:
            f.write(f'Truth: {truth}\n')
            for predname, pred_results in results_by_preds.items():
                f.write(f'vs preds: {predname}\n')
                for mname, mvalue in pred_results.items():
                    f.write(f'  * {mname} = {mvalue}\n')

def todo(*_):
    raise NotImplementedError


if __name__ == '__main__':
    main()
