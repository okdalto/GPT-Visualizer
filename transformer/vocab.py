import numpy as np

# Tiny Shakespeare character vocabulary (lowercased, minimal)
# 26 letters + space + 4 punctuation + PAD = 32
CHARS = list(" !'.?abcdefghijklmnopqrstuvwxyz")
SPECIAL = ["<PAD>"]
VOCAB = CHARS + SPECIAL
VOCAB_SIZE = len(VOCAB)  # 32
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
PAD_ID = CHAR_TO_ID["<PAD>"]  # 31


def encode_text(text: str, seq_len: int) -> np.ndarray:
    """Encode a text string into token IDs, truncated/padded to seq_len.
    Characters not in vocabulary are skipped."""
    ids = [CHAR_TO_ID[c] for c in text.lower().replace('\n', ' ') if c in CHAR_TO_ID]
    ids = ids[:seq_len]
    ids += [PAD_ID] * (seq_len - len(ids))
    return np.array(ids, dtype=np.int32)


def decode_ids(ids: np.ndarray) -> str:
    """Decode token IDs back to a string."""
    chars = []
    for i in ids:
        tok = ID_TO_CHAR.get(int(i), "?")
        if tok == "<PAD>":
            break
        chars.append(tok)
    return "".join(chars)


def make_labels(ids: np.ndarray) -> list[str]:
    """Create display labels for visualization."""
    labels = []
    for i in ids:
        tok = ID_TO_CHAR.get(int(i), "?")
        if tok == "<PAD>":
            labels.append("_")
        elif tok == " ":
            labels.append(" ")
        else:
            labels.append(tok)
    return labels
