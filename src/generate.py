import torch
from .data import indices_to_text
from .model import CharLSTM
import hashlib


def greedy_generate(model, start_symbol, length, itos, device):
    model.eval()
    hidden = model.init_hidden(1, device)
    idx = torch.tensor([[start_symbol]], device=device)
    result = [start_symbol]
    for _ in range(length - 1):
        logits, hidden = model(idx, hidden)
        probs = logits[-1, 0].softmax(dim=0)
        idx = torch.argmax(probs).view(1, 1)
        result.append(int(idx))
    return indices_to_text(result, itos)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_text(weights, input_path, output_path, hidden, layers):
    # Load raw for length
    raw = open(input_path, "rb").read()
    text = raw.decode("utf-8", errors="ignore")
    BOS = "\u241f"
    full = BOS + text
    length = len(full)

    # Load model
    checkpoint = torch.load(weights, map_location="cpu")
    stoi, itos = checkpoint["stoi"], checkpoint["itos"]
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = CharLSTM(len(stoi), hidden_size=hidden, num_layers=layers).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Generate
    out = greedy_generate(model, stoi[BOS], length, itos, device)[1:]
    with open(output_path, "wb") as f:
        f.write(out.encode("utf-8"))
    print("Generated and saved to", output_path)

    # Compare SHA256 hashes
    input_hash = sha256_file(input_path)
    output_hash = sha256_file(output_path)
    print(f"Input SHA256:  {input_hash}")
    print(f"Output SHA256: {output_hash}")
    if input_hash == output_hash:
        print("Hashes match.")
    else:
        print("Hashes do NOT match.")
