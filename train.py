import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

# =========================================
# Load Dataset
# =========================================

df = pd.read_csv("gk_qna_dataset.csv")

# =========================================
# Tokenization
# =========================================

def tokenize(text):

    # lowercase
    text = text.lower()

    # remove quotes
    text = re.sub(r"[\"']", "", text)

    # remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    return tokens

# =========================================
# Build Vocabulary
# =========================================

vocab = {"": 0}

def build_vocab(row):

    tokenized_question = tokenize(row["question"])
    tokenized_answer = tokenize(row["answer"])

    merged_tokens = tokenized_question + tokenized_answer

    for token in merged_tokens:

        if token not in vocab:
            vocab[token] = len(vocab)

df.apply(build_vocab, axis=1)

print(f"Vocabulary Size: {len(vocab)}")

# =========================================
# Text to Indices
# =========================================

def text_to_indices(text, vocab):

    indexed_text = []

    for token in tokenize(text):

        if token in vocab:
            indexed_text.append(vocab[token])

        else:
            indexed_text.append(vocab[""])

    return indexed_text

# =========================================
# Dataset
# =========================================

class QADataset(Dataset):

    def __init__(self, df, vocab):

        self.df = df
        self.vocab = vocab

    def __len__(self):

        return self.df.shape[0]

    def __getitem__(self, index):

        numerical_question = text_to_indices(
            self.df.iloc[index]["question"],
            self.vocab
        )

        numerical_answer = text_to_indices(
            self.df.iloc[index]["answer"],
            self.vocab
        )

        return (
            torch.tensor(numerical_question),
            torch.tensor(numerical_answer)
        )

# =========================================
# Dataloader
# =========================================

dataset = QADataset(df, vocab)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

# =========================================
# Model
# =========================================

class simpleRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        hidden_size=64
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, question):

        embedded = self.embedding(question)

        _, final = self.rnn(embedded)

        output = self.fc(final.squeeze(0))

        return output

# =========================================
# Initialize Model
# =========================================

model = simpleRNN(vocab_size=len(vocab))

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 50

# =========================================
# Training
# =========================================

print("Training Started...")

for epoch in range(epochs):

    total_loss = 0

    for question, answer in dataloader:

        optimizer.zero_grad()

        output = model(question)

        # only first answer token
        target = answer[0][0].unsqueeze(0)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}"
        )

print("Training Finished!")

# =========================================
# Create Reverse Vocabulary
# =========================================

idx_to_word = {
    idx: word
    for word, idx in vocab.items()
}

# =========================================
# Prediction Function
# =========================================

def predict(question_text):

    model.eval()

    with torch.no_grad():

        indices = text_to_indices(question_text, vocab)

        if not indices:
            return ""

        x = torch.tensor(indices).unsqueeze(0)

        pred = torch.argmax(model(x), dim=1).item()

    return idx_to_word.get(pred, "")

# =========================================
# Test Predictions
# =========================================

tests = [
    "What is the capital of France?",
    "What is H2O commonly called?",
    "Which planet is known as the red planet?",
    "Which bird cannot fly?",
    "What is the capital of Japan?"
]

print("\nTesting Predictions:\n")

for q in tests:

    print(f"{q:45} -> {predict(q)}")

# =========================================
# Save Model
# =========================================

torch.save(
    model.state_dict(),
    "model.pth"
)

# =========================================
# Save Vocabulary
# =========================================

with open("vocab.pkl", "wb") as f:

    pickle.dump(vocab, f)

with open("idx_to_word.pkl", "wb") as f:

    pickle.dump(idx_to_word, f)

print("\nModel and vocab saved successfully!")