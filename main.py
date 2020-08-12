import argparse

from utils.drive_manually import Game
from utils.training import Training
from utils.testing import Testing

# main fn
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive-manually', '-m', action='store_true', help="Make dataset from human expert’s demonstrations.")
    parser.add_argument('--training', '-t', action='store_true', help="Run a training process.")
    parser.add_argument('--testing', action='store_true', help="Run a testing process.")
    parser.add_argument('--dataset', '-d', type=str, help="Dataset file (.npz)")
    parser.add_argument('--hid', nargs='+', type=int, help="Set the shape of hidden layers", default=[32, 64])
    args = parser.parse_args()

    #print(args)

    # Run game
    if args.drive_manually == True:
        Game()
    # Run training
    if args.training == True:
        Training(hid=args.hid)
    # Run testings
    if args.testing == True:
        Testing()