import os
import json

filepath = os.path.abspath(__file__)
src_dataset_directory = os.path.dirname(filepath)
src_directory = os.path.dirname(src_dataset_directory)
root_directory = os.path.dirname(src_directory)
config_filepath = os.path.join(root_directory, "config.json")

with open(config_filepath, "r") as file:
    config = json.load(file)

models_directory = os.path.join(root_directory, config["models_directory"])
stats_directory = os.path.join(root_directory, config["stats_directory"])

word2vec_config = config["word2vec"]
