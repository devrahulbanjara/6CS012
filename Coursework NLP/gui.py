import pickle
import re
import torch
import torch.nn as nn
import gradio as gr
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]


class LSTMWord2VecClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        out = self.fc(hidden)
        return out


vocab_size = len(vocab)
model = LSTMWord2VecClassifier(
    vocab_size=vocab_size, embed_dim=300, hidden_dim=128, num_layers=2
)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()


def clean_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def text_to_indices(text, vocab, max_len):
    tokens = text.split()[:max_len]
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    padded = [0] * (max_len - len(indices)) + indices
    return padded


def predict(headline):
    cleaned = clean_text(headline)
    indices = text_to_indices(cleaned, vocab, max_len)
    x = torch.tensor([indices], dtype=torch.long)
    with torch.no_grad():
        out = model(x).squeeze(1)
        prob = torch.sigmoid(out).item()
    pred = "Sarcastic" if prob >= 0.5 else "Not Sarcastic"
    return pred, round(prob, 4)


iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter a headline..."),
    outputs=[gr.Textbox(label="Prediction"), gr.Textbox(label="Confidence")],
    title="Sarcasm Detector",
    description="Enter a news headline to check if it is sarcastic or not.",
)

if __name__ == "__main__":
    iface.launch()
