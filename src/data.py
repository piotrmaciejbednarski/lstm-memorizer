import hashlib


def build_vocab(text: str):
    """Builds char-level vocab mappings."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def text_to_indices(text: str, stoi: dict):
    return [stoi[ch] for ch in text]


def indices_to_text(indices: list, itos: dict) -> str:
    return "".join(itos[i] for i in indices)


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()
