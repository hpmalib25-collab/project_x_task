import torch
import torch.nn as nn

texts = []
labels = []

# Read dataset
with open("dataset.txt", "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        if "|" not in line:
            continue

        label, text = line.split("|", 1)

        labels.append(label)
        texts.append(text)

print("Number of articles:", len(texts))


# Tokenizer
def tokenize(text):
    return text.lower().split()


# Vocabulary
vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for text in texts:

    for word in tokenize(text):

        if word not in vocab:
            vocab[word] = len(vocab)

print("Vocabulary Size:", len(vocab))


# Encode text
def encode(text):

    return [
        vocab.get(word, vocab["<UNK>"])
        for word in tokenize(text)
    ]


encoded_texts = [
    encode(text)
    for text in texts
]


# Encode labels
label_map = {
    "Sports": 0,
    "Technology": 1,
    "Politics": 2
}

encoded_labels = [
    label_map[label]
    for label in labels
]

print("First 10 Labels:")
print(encoded_labels[:10])


# Maximum sequence length
max_len = max(len(seq) for seq in encoded_texts)

print("Max Length:", max_len)


# Padding
padded_texts = []

for seq in encoded_texts:

    padded_seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))

    padded_texts.append(padded_seq)


# Convert to tensors
X = torch.tensor(padded_texts)

y = torch.tensor(encoded_labels)

print("\nDataset Shapes")
print("X shape =", X.shape)
print("y shape =", y.shape)

print("\nFirst Article:")
print(X[0])

print("\nFirst Label:")
print(y[0])


class TransformerClassifier(nn.Module):

    def __init__(self, vocab_size, d_model, num_classes):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(1, 2))

        scores = scores / (Q.shape[-1] ** 0.5)

        weights = torch.softmax(scores, dim=-1)

        attention_output = torch.matmul(weights, V)

        pooled = attention_output.mean(dim=1)

        logits = self.classifier(pooled)

        return logits
    
model = TransformerClassifier(
    vocab_size=len(vocab),
    d_model=32,
    num_classes=3
)

print(model)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100

for epoch in range(epochs):

    logits = model(X)

    loss = loss_fn(logits, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:

        predictions = torch.argmax(logits, dim=1)

        accuracy = (predictions == y).float().mean()

        print(
            f"Epoch {epoch} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.4f}"
        )

test_article = "Virat Kohli scored a century in the final"

tokens = encode(test_article)

tokens = tokens[:max_len]

tokens += [0] * (max_len - len(tokens))

test_tensor = torch.tensor([tokens])

with torch.no_grad():

    output = model(test_tensor)

    prediction = torch.argmax(output, dim=1).item()

classes = [
    "Sports",
    "Technology",
    "Politics"
]

print("\nPrediction:")
print(classes[prediction])