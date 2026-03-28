"""
Traffic Sign Classifier — NLP Component (Scaffold / Prototype)
===============================================================
Demonstrates a TextCNN that classifies natural-language sign
descriptions back to GTSRB class IDs.

Components:
  1. GTSRB_DESCRIPTIONS — mapping of 43 class IDs → sign descriptions
  2. TextCNN — a small 1-D convolution text classifier
  3. Training loop on the synthetic sign-description dataset

This satisfies the NLP + CNN cross-requirement.

Run:
    python src/nlp_component.py
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from data_pipeline import load_config


# ==============================================================
# Sign Descriptions (43 GTSRB classes)
# ==============================================================

GTSRB_DESCRIPTIONS = {
    0:  "Speed limit twenty kilometers per hour zone ahead",
    1:  "Speed limit thirty kilometers per hour zone ahead",
    2:  "Speed limit fifty kilometers per hour zone ahead",
    3:  "Speed limit sixty kilometers per hour zone ahead",
    4:  "Speed limit seventy kilometers per hour zone ahead",
    5:  "Speed limit eighty kilometers per hour zone ahead",
    6:  "End of speed limit eighty kilometers per hour zone",
    7:  "Speed limit one hundred kilometers per hour zone ahead",
    8:  "Speed limit one hundred twenty kilometers per hour zone ahead",
    9:  "No passing zone for all vehicles",
    10: "No passing zone for vehicles over three point five metric tons",
    11: "Right of way at the next intersection priority",
    12: "Priority road ahead continue with caution",
    13: "Yield sign give way to oncoming traffic",
    14: "Stop sign vehicle must come to a full stop",
    15: "No vehicles allowed restricted entry zone",
    16: "Vehicles over three point five metric tons prohibited",
    17: "No entry do not enter this road",
    18: "General caution warning danger ahead proceed carefully",
    19: "Dangerous curve to the left warning ahead",
    20: "Dangerous curve to the right warning ahead",
    21: "Double curve warning first curve to the left",
    22: "Bumpy road surface warning uneven pavement ahead",
    23: "Slippery road surface warning when wet",
    24: "Road narrows on the right side ahead",
    25: "Road work construction zone ahead proceed slowly",
    26: "Traffic signals ahead prepare to stop at lights",
    27: "Pedestrians crossing zone ahead slow down",
    28: "Children crossing zone school area ahead slow down",
    29: "Bicycles crossing zone cyclists ahead be aware",
    30: "Beware of ice or snow on road surface ahead",
    31: "Wild animals crossing zone deer or wildlife ahead",
    32: "End of all speed and passing limits zone",
    33: "Turn right ahead mandatory direction sign",
    34: "Turn left ahead mandatory direction sign",
    35: "Ahead only go straight mandatory direction sign",
    36: "Go straight or turn right mandatory direction",
    37: "Go straight or turn left mandatory direction",
    38: "Keep right mandatory pass on the right side",
    39: "Keep left mandatory pass on the left side",
    40: "Roundabout mandatory circular traffic ahead",
    41: "End of no passing zone vehicles may pass",
    42: "End of no passing zone for heavy vehicles over three point five tons",
}


# ==============================================================
# Data augmentation — expand synthetic dataset
# ==============================================================

def augment_descriptions(descriptions: dict, copies: int = 15):
    """
    Create augmented variants of each sign description by:
      - Dropping random words
      - Shuffling word order slightly
      - Adding noise words
    Returns list of (text, label) pairs.
    """
    noise_words = ["warning", "sign", "road", "ahead", "caution",
                   "traffic", "zone", "area", "notice", "alert"]
    samples = []
    for label, desc in descriptions.items():
        words = desc.split()
        samples.append((desc, label))  # original
        for _ in range(copies):
            variant = words.copy()
            # Random word drop (10-30% of words)
            drop_n = max(1, int(len(variant) * random.uniform(0.1, 0.3)))
            for __ in range(drop_n):
                if len(variant) > 2:
                    idx = random.randint(0, len(variant) - 1)
                    variant.pop(idx)
            # Possibly add a noise word
            if random.random() > 0.5:
                variant.insert(random.randint(0, len(variant)),
                               random.choice(noise_words))
            # Slight shuffle (swap adjacent pairs)
            if random.random() > 0.5 and len(variant) > 2:
                i = random.randint(0, len(variant) - 2)
                variant[i], variant[i+1] = variant[i+1], variant[i]
            samples.append((" ".join(variant), label))
    random.shuffle(samples)
    return samples


# ==============================================================
# Vocabulary builder
# ==============================================================

class Vocabulary:
    """Simple word-level vocabulary."""

    def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.pad_idx = 0
        self.unk_idx = 1

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())
        for word, _ in counter.most_common():
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=20):
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(w, self.unk_idx) for w in tokens]
        # Pad
        ids += [self.pad_idx] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ==============================================================
# TextCNN Model
# ==============================================================

class TextCNN(nn.Module):
    """
    1-D CNN for text classification.
    Embeds words → Conv1D filters of multiple sizes → Max-pool → FC.
    """

    def __init__(self, vocab_size, embed_dim=64, num_filters=32,
                 filter_sizes=(2, 3, 4), num_classes=43, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (B, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)  # (B, embed_dim, seq_len)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(emb))              # (B, num_filters, L')
            c = c.max(dim=2).values                # (B, num_filters)
            conv_outs.append(c)
        out = torch.cat(conv_outs, dim=1)           # (B, num_filters * n)
        out = self.dropout(out)
        return self.fc(out)


# ==============================================================
# Training
# ==============================================================

def train_text_cnn(cfg):
    """Train TextCNN and print metrics."""
    nlp_cfg = cfg.get("nlp", {})
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60)
    print("  NLP COMPONENT — TextCNN on Sign Descriptions")
    print("=" * 60)

    # Build dataset
    samples = augment_descriptions(GTSRB_DESCRIPTIONS, copies=15)
    texts = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    vocab = Vocabulary()
    vocab.build(texts)
    print(f"  Vocabulary size : {len(vocab)}")
    print(f"  Total samples   : {len(samples)}")

    max_len = 20
    X = torch.tensor([vocab.encode(t, max_len) for t in texts])
    y = torch.tensor(labels)

    # 80/20 split
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Model
    embed_dim = nlp_cfg.get("embedding_dim", 64)
    num_filters = nlp_cfg.get("num_filters", 32)
    filter_sizes = tuple(nlp_cfg.get("filter_sizes", [2, 3, 4]))
    dropout = nlp_cfg.get("dropout", 0.3)
    epochs = nlp_cfg.get("epochs", 30)
    lr = nlp_cfg.get("learning_rate", 0.001)

    model = TextCNN(len(vocab), embed_dim, num_filters, filter_sizes,
                    num_classes=43, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_preds = val_logits.argmax(1)
                val_acc = (val_preds == y_val).float().mean().item()
                train_preds = logits.argmax(1)
                train_acc = (train_preds == y_train).float().mean().item()
            print(f"  Epoch {epoch:3d}/{epochs}  loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    # Final metrics
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_preds = val_logits.argmax(1).numpy()
        val_true = y_val.numpy()

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(val_true, val_preds)
    f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

    print(f"\n  NLP Val Accuracy : {acc:.4f}")
    print(f"  NLP Val Macro-F1 : {f1:.4f}")

    # Save metrics
    log_path = os.path.join(cfg["paths"]["logs"], "nlp_metrics.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"val_accuracy": round(acc, 4),
                   "val_macro_f1": round(f1, 4)}, f, indent=2)
    print(f"  NLP metrics saved → {log_path}")

    return model, vocab


# ==============================================================
# Utility: describe a sign given class ID
# ==============================================================

def describe_sign(class_id: int) -> str:
    """Return the natural-language description for a GTSRB class ID."""
    return GTSRB_DESCRIPTIONS.get(class_id, f"Unknown sign (class {class_id})")


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    cfg = load_config()
    model, vocab = train_text_cnn(cfg)

    # Quick demo
    print("\n--- Sign Description Demo ---")
    for cid in [0, 14, 25, 33, 40]:
        print(f"  Class {cid:2d}: {describe_sign(cid)}")
    print("\n✓ NLP component complete.")
