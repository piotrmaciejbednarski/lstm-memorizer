import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List
import json

from .model import CharLSTM
from .data import text_to_indices
from safetensors.torch import load_file


def plot_training_loss(
    loss_history: List[float], output_path: str = "training_loss.png"
):
    """Generate and save training loss plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2, color="#2E86AB")
    plt.title("Training Loss Over Time", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training loss plot saved to {output_path}")


def plot_hidden_states_heatmap(
    weights_path: str,
    input_path: str,
    hidden_size: int,
    num_layers: int,
    output_path: str = "hidden_states_heatmap.png",
    max_chars: int = 100,
):
    """Generate and save hidden states heatmap."""
    # Load model and vocab
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    weights_dir = Path(weights_path).parent
    vocab_path = weights_dir / "vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        stoi = vocab_data["stoi"]
        itos = vocab_data["itos"]

    # Load text
    raw = open(input_path, "rb").read()
    text = raw.decode("utf-8", errors="ignore")
    BOS = "\u241f"
    full = BOS + text

    # Prepare data
    data = text_to_indices(full[:max_chars], stoi)
    seq = torch.tensor(data, dtype=torch.long, device=device).unsqueeze(1)

    # Load model
    model = CharLSTM(len(stoi), hidden_size=hidden_size, num_layers=num_layers).to(
        device
    )
    model.load_state_dict(load_file(weights_path))
    model.eval()

    # Get hidden states
    hidden_states = model.init_hidden(batch_size=1, device=device)
    all_hidden_states = []

    with torch.no_grad():
        for i in range(len(seq) - 1):
            inp = seq[i : i + 1]
            _, hidden_states = model(inp, hidden_states)
            # Take the last layer's hidden state
            h_state = hidden_states[0][-1, 0, :].cpu().numpy()
            all_hidden_states.append(h_state)

    # Create heatmap
    hidden_matrix = np.array(all_hidden_states).T

    plt.figure(figsize=(12, 8))
    plt.imshow(hidden_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Activation Value")
    plt.title("LSTM Hidden States Heatmap", fontsize=16, fontweight="bold")
    plt.xlabel("Character Position", fontsize=12)
    plt.ylabel("Hidden Unit", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Hidden states heatmap saved to {output_path}")


def plot_attention_weights(
    weights_path: str,
    input_path: str,
    hidden_size: int,
    num_layers: int,
    output_path: str = "attention_heatmap.png",
    max_chars: int = 50,
):
    """Generate attention-like visualization from hidden states correlations."""
    # Load model and vocab
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    weights_dir = Path(weights_path).parent
    vocab_path = weights_dir / "vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        stoi = vocab_data["stoi"]
        itos = vocab_data["itos"]

    # Load text
    raw = open(input_path, "rb").read()
    text = raw.decode("utf-8", errors="ignore")
    BOS = "\u241f"
    full = BOS + text

    # Prepare data
    data = text_to_indices(full[:max_chars], stoi)
    seq = torch.tensor(data, dtype=torch.long, device=device).unsqueeze(1)

    # Load model
    model = CharLSTM(len(stoi), hidden_size=hidden_size, num_layers=num_layers).to(
        device
    )
    model.load_state_dict(load_file(weights_path))
    model.eval()

    # Get hidden states
    hidden_states = model.init_hidden(batch_size=1, device=device)
    all_hidden_states = []
    chars = []

    with torch.no_grad():
        for i in range(len(seq) - 1):
            inp = seq[i : i + 1]
            _, hidden_states = model(inp, hidden_states)
            # Take the last layer's hidden state
            h_state = hidden_states[0][-1, 0, :].cpu().numpy()
            all_hidden_states.append(h_state)
            chars.append(itos[str(inp.item())])

    # Calculate attention-like matrix (correlation between hidden states)
    hidden_matrix = np.array(all_hidden_states)
    attention_matrix = np.corrcoef(hidden_matrix)

    # Create heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title(
        "Hidden States Correlation Matrix\n(Attention-like Visualization)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Character Position", fontsize=12)
    plt.ylabel("Character Position", fontsize=12)

    # Add character labels if not too many
    if len(chars) <= 20:
        tick_labels = [c if c.isprintable() and c != " " else "·" for c in chars]
        plt.xticks(range(len(chars)), tick_labels, rotation=45)
        plt.yticks(range(len(chars)), tick_labels)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Attention-like heatmap saved to {output_path}")


def generate_demo_charts(
    weights_path: str,
    input_path: str,
    hidden_size: int,
    num_layers: int,
    output_dir: str = "charts",
):
    """Generate all demonstration charts."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate synthetic training loss for demo
    epochs = 4000
    loss_history = []
    initial_loss = 4.0
    final_loss = 0.006

    for i in range(epochs):
        # Exponential decay with some noise
        progress = i / epochs
        loss = (
            initial_loss * np.exp(-5 * progress)
            + final_loss
            + np.random.normal(0, 0.01)
        )
        loss = max(loss, 0.001)  # Ensure non-negative
        loss_history.append(loss)

    plot_training_loss(loss_history, str(output_path / "training_loss.png"))
    plot_hidden_states_heatmap(
        weights_path,
        input_path,
        hidden_size,
        num_layers,
        str(output_path / "hidden_states_heatmap.png"),
    )
    plot_attention_weights(
        weights_path,
        input_path,
        hidden_size,
        num_layers,
        str(output_path / "attention_heatmap.png"),
    )

    print(f"All demo charts generated in {output_dir}/")
