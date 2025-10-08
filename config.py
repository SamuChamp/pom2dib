import argparse
from utils.utils import add_flags_from_config


"""
if args.no_det == True and args.sampling == 'opt': args.method = 'pom2dib'
elif args.no_det == True and args.sampling == 'rand': args.method = 'rsdib'
elif args.no_det == True and args.sampling == 'full': args.method = 'tadib'
elif args.no_det == False and args.sampling == 'full': args.method = 'dlsc'
"""


config_args = {
    'data_config': {
        'dataset': ('mmfi', 'handwritten or mmfi'),
    },
    'model_config': {
        'no_det': (True, 'deterministic encoder or not'),
        'num_of_sel_Tx': (4, 'number of transmitter_side selection'),
        'num_of_sel_Rx': (2, 'number of receiver_side selection'),
        'net_en': (['Base', 'Base'], 'MobileNet or Base, which net_structure to use for encoder & its selector'),
        'net_de': (['Base', 'Base'], 'which net_structure to use for decoder & its selector'),
        'embd_dim': (24, 'embedding dimension'),
    },
    'training_config': {
        'sampling': ('opt', 'selection policy: opt or rand or full'),
        'sparse': (False, 'sparse selection or not'),
        'gamma': (1e-3, 'sparse selection coefficient'),
        'beta': (1e-3, 'bottleneck coefficient'),
        'bs': (20, 'batch size'),
        'lr_code': (1e-4, 'learning rate for en/de coder'),
        'lr_sel': (0.5e-4, 'learning rate for selector'),
        'epochs': (2000, 'number of training epochs'),
        'seed': (114514, 'seed for training'),
        'log_freq': (5, 'printing frequency of train/val metrics (in epochs)'),
    },
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
    