import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Load spaCy tokenizer
nlp = spacy.load("en_core_web_sm")

PAD_IDX = 0
UNK_IDX = 1

def tokenize(text, prt=False):
    if prt:
        arr = []
        print(text)
        print("----------------------------------------")
        for token in nlp(str(text)):
            # print('token: ',token)
            arr.append(token.text.lower())
        return arr
    else:
        return [token.text.lower() for token in nlp(text)]

# Build vocabulary
def build_vocab(texts, max_vocab_size=25_000):
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for text in texts:
        for token in tokenize(text):
            vocab
        
    counter = Counter(token for text in texts for token in tokenize(text))
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
    vocab["<pad>"] = PAD_IDX
    vocab["<unk>"] = UNK_IDX
    return vocab

# Numericalize text
def numericalize(texts, vocab):
    return [[vocab.get(token, UNK_IDX) for token in tokenize(text, True)] for text in texts]

# Load IMDB dataset
def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ["pos", "neg"]:
        folder = f"{data_dir}/{label_type}"
        for file in os.listdir(folder):
            with open(f"{folder}/{file}", "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(1 if label_type == "pos" else 0)
    return texts, labels


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = numericalize(texts, vocab)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        print('idx: ', idx)
        print('self.texts[idx]: ', self.texts[idx])
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.float)
    return padded_texts, labels, lengths


class LogisticRegression(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: [batch_size, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)  # Average over the sequence length
        return self.fc(pooled).squeeze(1)  # Shape: [batch_size]


def train_model(model, dataloader, optimizer, criterion, device):
    print('train start')
    model.train()
    epoch_loss = 0
    for texts, labels, _ in tqdm(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('train end')
    return epoch_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels, _ in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            all_preds.extend(torch.round(torch.sigmoid(predictions)).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(dataloader), accuracy


# Load data
import os
print('load dataset')
train_texts, train_labels = load_imdb_data("./aclImdb/train")
test_texts, test_labels = load_imdb_data("./aclImdb/test")
print(len(train_texts))
print(len(train_labels))
print(len(test_texts))
print(len(test_labels))
test_texts = test_texts[:1000]
test_labels = test_labels[:1000]



# Split data
# train_texts, valid_texts = train_texts[:20000], train_texts[20000:]
# train_labels, valid_labels = train_labels[:20000], train_labels[20000:]
train_texts, train_labels, valid_texts, valid_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=1234)
train_texts = train_texts[:1000]
train_labels = train_labels[:1000]
valid_texts = valid_texts[1000:1200]
valid_labels = valid_labels[1000:1200]
# Build vocabulary
print('build vocab')
vocab = build_vocab(train_texts)

# Create datasets and dataloaders
print('dataset')
train_dataset = IMDBDataset(train_texts, train_labels, vocab)
valid_dataset = IMDBDataset(valid_texts, valid_labels, vocab)
test_dataset = IMDBDataset(test_texts, test_labels, vocab)

print('dataloader')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)



# Model, optimizer, and loss function
print('init model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LogisticRegression(len(vocab), embed_dim=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Train the model
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    print('epoch')
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}, Valid Accuracy = {valid_acc:.6f}")