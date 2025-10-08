import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config.
    """
    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object.
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str.
            elif default is None:
                return str(x)
            # Special case: if default is a boolean, use str2bool for parsing.
            elif isinstance(default, bool):
                return str2bool(x)
            # Otherwise, convert x to the type of the default.
            else:
                return type(default)(x)
        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument.
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                        help=description
                    )
                else:
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        default=default,
                        help=description
                    )
            else:
                parser.add_argument(
                    f"--{param}",
                    type=OrNone(default),
                    default=default,
                    help=description
                )
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

if __name__ == "__main__":
    pass