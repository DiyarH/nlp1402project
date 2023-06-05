# Text Topic Classification Project
This is the project for the 1402 Natural Language Processing course at IUST.
## Depenedencies
* Python 3 (Tested on Python 3.11)
* tqdm
* remotezip
* pandas
* datasets
* matplotlib
* ijson
## Dataset
### Quickstart
The easiest way to access the dataset is via HuggingFace.
To load the dataset, first install the `datasets` package:
```
python -m pip install datasets
```
Then, load the dataset in your python code:
```
from datasets import load_dataset
dataset = load_dataset('diyarhamedi/HowTo100M-subtitles-small')
```
### Collecting the dataset from scratch
If you wish to collect the raw data and process it by yourself, follow these steps:
1. Clone this repository:
```
git clone https://github.com/DiyarH/nlp1402project
```
2. Create a virtual environment in the root directory and install the required packages:
```
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```
3. Change the configuration (Optional)

You can customize the data creation process by changing the `config.json` file:
* `"paths"`: Directory to which the data is saved.
* `"urls"`: The URL to the remote data files.
You may change this if the files are moved,
or if you already store the files on a remote URL.
* `"chunksize"`: The chunk sizes used by each step for loading large data.
The `"collector"` option is in Kilobytes, and the rest are in number of items times 1024.
* `"sampling_seed"`: The seed used for sampling from the large data.
You may change this if you want to achieve a different set of samples for your requirements.
* `"max_n_samples_per_category"`: Specify the maximum amount of samples from each category.
* `"stats_n_top_words"`: Specify how many of the top words for each of the following metrics are saved:
  * Local Frequency : per category
  * RNF (Relative Normalized Frequency) : per category
  * TF-IDF (Term Frequency - Inverse Document Frequency) : per category
  * Total Frequency : all words
4. Collect and process the data
Run the entire process:
```
cd src && python dataset
```
This will automatically collect, categorize, sample and cleanup the raw data, break the result into sentences and words, and extract useful linguistic metrics for data analysis.

CAUTION: This will downlaod and process nearly 7.5GB of data,
so it is advised to try this option in an environment with atleast 15GB of free space and adequate download rate.

Alternatively, you can run each step separately by running the corresponding files in order:
```
cd src
python dataset/collect.py     # Download the data files
python dataset/categorize.py  # Categorize the downloaded data
python dataset/cleanup.py     # Sample and cleanup the raw data
python dataset/sent_break.py  # Break the dataset corpus by sentence
python dataset/word_break.py  # Break the dataset corpus by word
python dataset/stats.py       # Extract and save linguistic metrics
```
5. Create dataset object from the files (Optional)

```
# `src.dataset` if you run the code from the root directory
# `dataset` if you run the code from the src/ directory
from dataset import load_dataset_from_files

loaded_dataset = load_dataset_from_files()
```