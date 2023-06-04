# Text Topic Classification Project
This is the project for the 1402 Natural Language Processing course at IUST.

## How to use the dataset
### Using HuggingFace
The recommended way to get the dataset is by running the following code in python and loading it from HuggingFace:
```
from datasets import load_dataset
dataset = load_dataset('diyarhamedi/HowTo100M-subtitles-small')
```
### Collecting the dataset from scratch
TODO: Package requirements are not added yet.

If you wish to collect the raw data and process it by yourself, clone this repository, install the required packages and run the following command in its root directory:
```
python src/dataset
```
This will collect, categorize, sample and cleanup the raw data, break the result into sentences and words, and extract useful linguistic metrics for data analysis.

WARNING: This will downlaod and process nearly 7.5GB of data,
so it is advised to try this option in an environment with atleast 15GB of free space and adequate download rate.
