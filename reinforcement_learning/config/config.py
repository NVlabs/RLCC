import argparse
import os
import sys

import yaml

ROOT_PATH = r'/swgwork/bfuhrer/projects/rlcc/new_simulator/rl-cc-demo/reinforcement_learning/'
sys.path.append(ROOT_PATH)

class Config(dict):
    def __init__(self, name='default', root_path='config', override=None, d=None):
        dict.__init__(self)

        if d:
            self._load_from_dict(d)
        else:
            # path = os.path.join('.', root_path, f'{name}.yaml') #FIXME
            path = os.path.join(ROOT_PATH, root_path, f'{name}.yaml') #FIXME
            self._load_config(path)
            if override is not None:
                Config._override_config(self, override)

    def save_partial(self, path, keys=None, **kwargs):
        if keys:
            dict_file = dict((k, self[k]) for k in keys)
        else:
            dict_file = dict(self)
        dict_file.update(kwargs)

        with open(os.path.join(path), 'w') as f:
            yaml.dump(dict_file, f, default_flow_style=False)

    def _load_config(self, path):
        with open(path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self._load_from_dict(data)

    def _load_from_dict(self, d):
        for key, val in d.items():
            if type(val) == dict:
                val = Config(d=val)
            self[key] = val

    @staticmethod
    def _override_config(config, args):
        for key in config.keys():
            if type(config[key]) == Config:
                Config._override_config(config[key], args)
            elif key in args.keys():
                val = args[key]
                if val and val != "None" and val != -1:
                    config[key] = val

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<Config ' + dict.__repr__(self) + '>'


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
