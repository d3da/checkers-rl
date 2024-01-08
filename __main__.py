import argparse
import pathlib
from plot import plot_hyperparameter_search, plot_train_history

from train import QModelTrainRun, VModelTrainRun


parser = argparse.ArgumentParser(description='Train or evaluate a Reinforcement Learning checkers AI')
# parser.set_defaults(action='')
subcommand_parser = parser.add_subparsers(dest='subcommand', required=True, help=None)

### Global parameters
# (none for now)

### Run hyperparameter search
hparam_search_parser = subcommand_parser.add_parser('hparam_search', help='Search for the best hyperparameters (default)')
hparam_search_parser.add_argument('--model', choices=['q', 'v'], required=True, help='Whether to use the state-action value (Q) model or state-value (V) model')
# TODO override the default settings for hyperparameter search (num_iterations, etc.)

### Train a (single) model
train_parser = subcommand_parser.add_parser('train', help='Train a model given hyperparameters')
train_parser.add_argument('--model', choices=['q', 'v'], required=True, help='Whether to use the state-action value (Q) model or state-value (V) model')
train_parser.add_argument('--plot', help='Plot', action='store_true')

### Evaluate a models performance
eval_parser = subcommand_parser.add_parser('eval', help='Evaluate a trained models performance')
eval_parser.add_argument('--model', choices=['q', 'v'], required=True, help='Whether to use the state-action value (Q) model or state-value (V) model')
eval_parser.add_argument('--vs', choices=['random', 'heuristics'], help='Enemy to evaluate against')

### Plot training or hyperparameter search
plot_parser = subcommand_parser.add_parser('plot', help='Plot training history or hyperparameter search')
plot_parser.add_argument('--type', choices=['train_hist', 'hparam_search'], required=True)
plot_parser.add_argument('path', type=pathlib.Path, help='Path of the plotted csv file')


# Parse the arguments in program call
args = parser.parse_args()
# parser.print_help()
print(args)


if args.subcommand == 'hparam_search':
    # TODO
    pass

elif args.subcommand == 'train':
    # TODO
    if args.plot:
        plot_train_history('???')

elif args.subcommand == 'eval':
    # TODO
    pass

elif args.subcommand == 'plot':
    if args.type == 'train_hist':
        plot_train_history(args.path)
    elif args.type == 'hparam_search':
        plot_hyperparameter_search(args.path)
    else:
        raise ValueError

else:
    parser.print_help()

exit()
