import argparse

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to palletize boxes in a bin""")
    parser.add_argument("--render", '-r', action='store_true', default=False,
                        help="Render the environment")
    parser.add_argument("--epochs", '-e', type=int, default=100)

    args = parser.parse_args()
    return args