import torch
import torch.nn as nn
import gradio as gr
import pickle
import re

# =========================================
# Load Vocabulary
# =========================================

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

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
# Text To Indices
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
# RNN Model
# =========================================

class simpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, question):
        embedded = self.embedding(question)
        _, final = self.rnn(embedded)
        output = self.fc(final.squeeze(0))
        return output

# =========================================
# Load Model
# =========================================

model = simpleRNN(vocab_size=len(vocab))

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# =========================================
# Prediction Function
# =========================================

def predict(question_text):

    model.eval()

    with torch.no_grad():

        indices = text_to_indices(
            question_text,
            vocab
        )

        if not indices:
            return "No prediction"

        x = torch.tensor(indices).unsqueeze(0)

        pred = torch.argmax(
            model(x),
            dim=1
        ).item()

    return idx_to_word.get(pred, "Unknown")

# =========================================
# Gradio UI
# =========================================

title = "Simple GK Question Answering System using RNN"

description = """
A beginner NLP project built using PyTorch, Simple RNN, and Gradio.

Example Questions:
- What is the capital of France?
- Which planet is known as the red planet?
- What is H2O commonly called?
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a General Knowledge question..."
    ),
    outputs=gr.Textbox(label="Predicted Answer"),
    title=title,
    description=description
)

# =========================================
# Launch App
# =========================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)