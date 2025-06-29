from pathlib import Path
import sys
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from .data import build_vocab, text_to_indices
from .model import CharLSTM

from safetensors.torch import save_file
import json


def train_model(input_path, epochs, lr, hidden, layers, weights_out):
    # Load text
    raw = open(input_path, "rb").read()
    text = raw.decode("utf-8", errors="ignore")
    BOS = "\u241f"
    full = BOS + text

    # Vocab and data
    stoi, itos = build_vocab(full)
    data = text_to_indices(full, stoi)

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}", file=sys.stderr)

    # Model, optimizer, loss
    model = CharLSTM(len(stoi), hidden_size=hidden, num_layers=layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    seq = torch.tensor(data, dtype=torch.long, device=device).unsqueeze(1)
    for ep in range(1, epochs + 1):
        model.train()
        hidden_states = model.init_hidden(batch_size=1, device=device)
        optimizer.zero_grad()
        inp = seq[:-1]
        tgt = seq[1:].view(-1)
        logits, _ = model(inp, hidden_states)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt)
        loss.backward()
        optimizer.step()
        if ep == 1 or ep % 500 == 0:
            print(f"Epoch {ep}/{epochs} loss={loss.item():.4f}", file=sys.stderr)

    # Save model weights
    save_file(model.state_dict(), weights_out)

    weights_path = Path(weights_out).absolute().parent
    vocab_path = weights_path / "vocab.json"

    # Save vocabulary
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False)

    print(f"Model saved to {weights_out}", file=sys.stderr)
