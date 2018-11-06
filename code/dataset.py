import argparse
import json


def read_data(filename):
    with open(filename) as f:
        data = json.load(f)

    return data


def main(args):
    data = read_data(args.file)
    print(data[0].keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to file of the dataset.")
    args = parser.parse_args()

    main(args)
