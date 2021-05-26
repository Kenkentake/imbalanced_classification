import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument("--cfg_file", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    return args
