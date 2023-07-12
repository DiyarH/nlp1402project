import os, sys
filepath = os.path.abspath(__file__)
filepath = os.path.dirname(filepath)
filepath = os.path.dirname(filepath)
sys.path.append(filepath)

from features import feature_functions, HowTo100MSubtitleFeaturesDataset
from tqdm.auto import tqdm
from architectures import choose_suitable_architecture
from dataset import load_dataset_from_huggingface
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


def change_features_device(features, device):
    for feature_name, feature in features.items():
        features[feature_name] = feature.to(device)


def train(model, train_dataset, val_dataset=None, n_epochs=3):
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=8)
    loss_function = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    history = {
        'train_loss': [],
        'train_accuracy': [],
    }
    if val_dataset is not None:
        history['val_loss'] = []
        history['val_accuracy'] = []

    for epoch in n_epochs:
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        pbar = tqdm(desc=f'Epoch {epoch}')

        model.train()
        for idx, (batch_features, batch_labels) in enumerate(train_dataloader):
            batch_features = change_features_device(batch_features)
            if len(batch_features) == 1:
                logits = model(batch_features.values()[0])
            else:
                logits = model(batch_features)

            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            avg_loss = running_loss / (idx+1)
            pbar.set_postfix(train_loss=avg_loss)
            history['train_loss'].append(avg_loss)

            predictions = torch.argmax(logits, dim=-1)
            correct = torch.sum(predictions == batch_labels)
            n_samples = batch_labels.size(0)
            running_correct += correct
            running_samples += n_samples
            avg_accuracy = running_correct / running_samples
            pbar.set_postfix(train_accuracy=avg_accuracy)
            history['train_accuracy'].append(avg_accuracy)
            pbar.update(1)
        
        if val_dataset is None:
            continue

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        model.eval()
        for idx, (batch_features, batch_labels) in enumerate(val_dataloader):
            batch_features = change_features_device(batch_features)
            if len(batch_features) == 1:
                logits = model(batch_features.values()[0])
            else:
                logits = model(batch_features)

            loss = loss_function(logits, batch_labels)
            running_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            correct = torch.sum(predictions == batch_labels)
            n_samples = batch_labels.size(0)
            running_correct += correct
            running_samples += n_samples
        
        val_loss = running_loss / (idx+1)
        pbar.set_postfix(val_loss=val_loss)
        history['val_loss'].append(val_loss)

        val_accuracy = running_correct / running_samples
        pbar.set_postfix(val_accuracy=val_accuracy)
        history['val_accuracy'].append(val_accuracy)

    return history


def run():
    dataset = load_dataset_from_huggingface()
    dataset = dataset['train']

    categories = set(dataset['category'])
    category2idx = {category: idx for idx, category in enumerate(categories)}
    num_classes = len(categories)

    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, 1 - (train_size + test_size), test_size]
    )

    for feature_name in feature_functions:
        train_features_dataset = HowTo100MSubtitleFeaturesDataset(train_dataset)
        val_features_dataset = HowTo100MSubtitleFeaturesDataset(val_dataset)
        test_features_dataset = HowTo100MSubtitleFeaturesDataset(test_dataset)

        features_shape = train_features_dataset.features.values()[0][0].shape
        model = choose_suitable_architecture(num_classes, features_shape)
        history = train(model, train_features_dataset, val_features_dataset)

if __name__ == '__main__':
    run()