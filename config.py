import os
import yaml

class BinConfig(object):

    #yamlファイルの読み込み
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}
        try:
            with open(config_path) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print("Wrong file or file path")

    def update(self, key, value):
        self.config[key] = value
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, stream=f,
                        default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print("Wrong file or file path")
            
    def keys(self):
        return self.config.keys()