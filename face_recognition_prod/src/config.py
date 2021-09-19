import yaml


def get_algorithm_params(key):
    with open("src/algorithms.yaml", "r") as stream:
        params = yaml.load(stream)
        return params[key]