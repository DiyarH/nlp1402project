import os, sys

filepath = os.path.abspath(__file__)
filepath = os.path.dirname(filepath)
filepath = os.path.dirname(filepath)
sys.path.append(filepath)

from dataset import load_dataset_from_huggingface
from word2vec.config import models_directory
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

language_model_training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="no",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=10,
    weight_decay=0.05,
    report_to="none",
)


class HowTo100MSubtitlesLMDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for sample in corpus:
            encodings_dict = tokenizer(
                "<CLS>" + sample + "<END>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def create_language_model_dataset(corpus, tokenizer):
    max_length = 1024
    lm_dataset = HowTo100MSubtitlesLMDataset(corpus, tokenizer, max_length)
    train_size = int(0.9 * len(lm_dataset))
    train_dataset, val_dataset = random_split(
        lm_dataset, [train_size, len(lm_dataset) - train_size]
    )
    return {"train": train_dataset, "val": val_dataset}


def finetune_gpt2(dataset, tokenizer):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.resize_token_embeddings(len(tokenizer))
    Trainer(
        model=model,
        args=language_model_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=lambda data: {
            "input_ids": torch.stack([f[0] for f in data]),
            "attention_mask": torch.stack([f[1] for f in data]),
            "labels": torch.stack([f[0] for f in data]),
        },
    ).train()
    return model


def run():
    dataset = load_dataset_from_huggingface()
    dataset = dataset["train"]

    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2", bos_token="<CLS>", eos_token="<END>", pad_token="<PAD>"
    )

    categories = list(set(dataset["category"]))
    for category in categories:
        categorized_corpus = [
            sample["text"]
            for sample in dataset.to_list()
            if sample["category"] == category
        ]

        lm_dataset = create_language_model_dataset(categorized_corpus, tokenizer)
        language_model = finetune_gpt2(dataset, tokenizer)

        category_name = category.lower().replace(" ", "_").replace("&", "and")
        filename = "{}.language_model".format(category_name)
        filepath = os.path.join(models_directory, filename)
        torch.save(language_model, filepath)

if __name__ == '__main__':
    run()