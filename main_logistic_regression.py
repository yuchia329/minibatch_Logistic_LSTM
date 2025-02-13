import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter
import random
import numpy as np
from tqdm import tqdm  # For loading bars
import time  # Added to track execution time
from sklearn.metrics import accuracy_score
import tarfile
from spacy.cli import download
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
import json
import os.path


# Ensure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Constants
PAD_IDX = 0
UNK_IDX = 1

# Tokenization function
def tokenize(text):
    return [token.text.lower() for token in nlp(text)]

# Build vocabulary with a progress bar
def build_vocab(texts, max_vocab_size=25_000):
    if os.path.exists('vocab.json'):
        with open('vocab.json', 'r') as f:
            vocab = json.load(f)
    else:
        counter = Counter()
        for text in tqdm(texts, desc="Building Vocabulary"):
            counter.update(tokenize(text))
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
        vocab["<pad>"] = PAD_IDX
        vocab["<unk>"] = UNK_IDX
        with open('vocab.json', 'w') as f:
            json.dump(vocab, f)
    return vocab

# Numericalize text with a progress bar
def numericalize(texts, vocab, cache_file):
    if os.path.exists(f'{cache_file}.json'):
        with open(f'{cache_file}.json', 'r') as f:
            numericalized_texts = json.load(f)
    else:
        numericalized_texts = []
        for text in tqdm(texts, desc="Numericalizing Texts"):
            numericalized_texts.append([vocab.get(token, UNK_IDX) for token in tokenize(text)])
        with open(f'{cache_file}.json', 'w') as f:
            json.dump(numericalized_texts, f)
    return numericalized_texts

# Load IMDB dataset with a progress bar
def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ["pos", "neg"]:
        folder = os.path.join(data_dir, label_type)
        for file in tqdm(os.listdir(folder), desc=f"Loading {label_type} Data"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(1 if label_type == "pos" else 0)
    return texts, labels

# IMDB Dataset class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, cache_file):
        self.texts = numericalize(texts, vocab, cache_file)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# Collate function for DataLoader
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_texts, labels, lengths.to(torch.long)

# Logistic Regression model
class LogisticRegression(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: [batch_size, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)  # Average over the sequence length
        return self.fc(pooled).squeeze(1)  # Shape: [batch_size]

# Training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    loop = tqdm(dataloader, desc="Training", leave=True)
    for texts, labels, _ in loop:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    epoch_time = time.time() - start_time
    return epoch_loss / len(dataloader), epoch_time

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []

    loop = tqdm(dataloader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for texts, labels, _ in loop:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            # Convert predictions to binary labels (0 or 1)
            pred_labels = torch.round(torch.sigmoid(predictions)).cpu().numpy()
            
            all_preds.extend(pred_labels)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    
    return epoch_loss / len(dataloader), accuracy, all_preds, all_labels


# Download and extract IMDB dataset
def download_and_extract_data():
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = "aclImdb_v1.tar.gz"
    if not os.path.exists(tar_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, tar_path)
    print("Extracting dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall()

# Main script
if __name__ == "__main__":
    download_and_extract_data()

    # Load data
    print("Loading training and testing data...")
    train_texts, train_labels = load_imdb_data("./aclImdb/train")
    test_texts, test_labels = load_imdb_data("./aclImdb/test")

    # Shuffle dataset using fixed random seed
    print("Shuffling and splitting data into training and validation sets...")
    combined = list(zip(train_texts, train_labels))
    random.Random(SEED).shuffle(combined)  # Use fixed seed for reproducibility
    train_texts, train_labels = zip(*combined)  # Unzip back into separate lists

    # Convert back to lists after shuffling
    train_texts, train_labels = list(train_texts), list(train_labels)

    # Split into training and validation sets
    train_texts, valid_texts = train_texts[:20000], train_texts[20000:]
    train_labels, valid_labels = train_labels[:20000], train_labels[20000:]

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_texts)

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, 'train')
    valid_dataset = IMDBDataset(valid_texts, valid_labels, vocab, 'valid')
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, 'test')

    # Define batch sizes to test
    # batch_sizes = [1, 8, 16, 32, 64, 128]
    batch_sizes = [1]
    training_times = []
    validation_accuracies = []

    # Grid search over learning rates
    best_lr = None
    best_valid_acc = 0
    results = {}
    # learning_rates = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    learning_rates = [1e-3]

    N_EPOCHS = 10

    # Select the device and criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()

    for lr in learning_rates:
        print(f"\nTraining with learning rate {lr} and batch size {batch_sizes[0]}...")
        
        model = LogisticRegression(len(vocab), embed_dim=100).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Loop through different batch sizes
        for batch_size in batch_sizes:
            print(f"\nTraining with batch size {batch_sizes[0]}...")

            # Create new DataLoaders for each batch size
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            # Train the model for each batch size
            total_start_time = time.time()
            total_epoch_time = 0
            for epoch in range(N_EPOCHS):
                print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
                train_loss, epoch_time = train_model(model, train_loader, optimizer, criterion, device)
                valid_loss, valid_acc, all_pred, all_labels = evaluate_model(model, valid_loader, criterion, device)

                total_epoch_time += epoch_time  # Sum epoch time
                print(f"Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}, Valid Accuracy = {valid_acc:.4f}")
                print(f"Epoch Time: {epoch_time:.2f} seconds")

            total_training_time = time.time() - total_start_time  # Compute total training time

            # Store results
            results[lr] = (total_training_time, valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_lr = lr

            print(f"\nLearning Rate {lr} - Total Training Time: {total_training_time:.2f} seconds")
            print(f"Learning Rate {lr} - Average Time Per Epoch: {total_epoch_time / N_EPOCHS:.2f} seconds")

            # Store results for plotting
            training_times.append(total_training_time)
            validation_accuracies.append(valid_acc)

            print(f"\nBatch Size {batch_size} - Total Training Time: {total_training_time:.2f} seconds")
            print(f"Batch Size {batch_size} - Average Time Per Epoch: {total_epoch_time / N_EPOCHS:.2f} seconds")

            # Evaluate the model on the test set
            #test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            #print(f"\nTest Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
            # Evaluate the best model on dev and test sets and save predictions
            print("\nEvaluating Best Model...")

            dev_loss, dev_acc, dev_preds, dev_labels = evaluate_model(model, valid_loader, criterion, device)
            test_loss, test_acc, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)

            # Save predictions to CSV files for analysis
            dev_results_df = pd.DataFrame({
                "Gold Label": dev_labels,
                "Predicted Label": dev_preds
            })
            dev_results_df.to_csv("dev_predictions_LR.csv", index=False)

            test_results_df = pd.DataFrame({
                "Gold Label": test_labels,
                "Predicted Label": test_preds
            })
            test_results_df.to_csv("test_predictions_LR.csv", index=False)

            print("Predictions saved to dev_predictions_LR.csv and test_predictions_LR.csv")

            # Print a sample of the predictions for review
            print("\nSample Predictions (Dev Set)")
            print(dev_results_df.head(10))  # Show first 10 predictions

            print("\nSample Predictions (Test Set)")
            print(test_results_df.head(10))  # Show first 10 predictions
        
        # """

        # Plot training time vs. batch size
        plt.figure(figsize=(8, 5))
        plt.plot(batch_sizes, training_times, marker='o', linestyle='-', label='Training Time')
        plt.xlabel('Batch Size')
        plt.ylabel('Total Training Time (seconds)')
        plt.title('Training Time vs. Batch Size')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("training_time_vs_batch_size_LR.png")  # Save plot
        print("Saved training time plot as training_time_vs_batch_size_LR.png")


        # Plot validation accuracy vs. batch size
        plt.figure(figsize=(8, 5))
        plt.plot(batch_sizes, validation_accuracies, marker='o', linestyle='-', color='red', label='Validation Accuracy')
        plt.xlabel('Batch Size')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs. Batch Size')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("validation_accuracy_vs_batch_size_LR.png")  # Save plot
        print("Saved validation accuracy plot as validation_accuracy_vs_batch_size.png_LR")

        # """
    
    # Plot validation accuracy vs. learning rate
    plt.figure(figsize=(8, 5))
    lrs = list(results.keys())
    valid_accuracies = [results[lr][1] for lr in lrs]
    plt.plot(lrs, valid_accuracies, marker='o', linestyle='-', color='red', label='Validation Accuracy')
    plt.xscale("log")  # Log scale for better visualization
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Learning Rate')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("validation_accuracy_vs_learning_rate_LR.png")  # Save plot
    print("Saved validation accuracy plot as validation_accuracy_vs_learning_rate_LR.png")

    # Plot training time vs. learning rate
    plt.figure(figsize=(8, 5))
    train_times = [results[lr][0] for lr in lrs]
    plt.plot(lrs, train_times, marker='o', linestyle='-', color='blue', label='Training Time')
    plt.xscale("log")  # Log scale for better visualization
    plt.xlabel('Learning Rate')
    plt.ylabel('Total Training Time (seconds)')
    plt.title('Training Time vs. Learning Rate')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("training_time_vs_learning_rate_LR.png")  # Save plot
    print("Saved training time plot as training_time_vs_learning_rate_LR.png")

    print(f"\nBest Learning Rate: {best_lr} with Validation Accuracy: {best_valid_acc:.4f}")