#!/usr/bin/env python3
"""Embed nfcorpus corpus+queries using all-MiniLM-L6-v2 ONNX model → .fbin"""

import json
import struct
import sys
import os
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "all-MiniLM-L6-v2.onnx")
VOCAB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "models", "minilm-l6-v2-vocab.txt")
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "nfcorpus")
OUTPUT_DIR = DATA_DIR

BERT_CLS = 101
BERT_SEP = 102
BERT_PAD = 0
BERT_UNK = 100
MAX_SEQ = 128


def load_vocab(path):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            token = line.strip()
            if token:
                vocab[token] = i
    return vocab


def load_jsonl(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "") or obj.get("title", "")
            texts.append(text)
    return texts


def _is_cjk(r):
    r = ord(r) if isinstance(r, str) else r
    return (0x4E00 <= r <= 0x9FFF or 0x3400 <= r <= 0x4DBF or
            0x20000 <= r <= 0x2A6DF or 0xF900 <= r <= 0xFAFF or
            0x2F800 <= r <= 0x2FA1F)


def _is_punct(r):
    r = ord(r) if isinstance(r, str) else r
    return 33 <= r <= 47 or 58 <= r <= 64 or 91 <= r <= 96 or 123 <= r <= 126


def normalize_text(text):
    text = text.lower()
    # Strip combining marks (accents)
    import unicodedata
    text = unicodedata.normalize("NFD", text)
    chars = []
    for r in text:
        if unicodedata.category(r) == "Mn":
            continue
        r_ord = ord(r)
        if _is_cjk(r_ord) or _is_punct(r_ord):
            chars.extend([" ", r, " "])
        elif r.isspace():
            chars.append(" ")
        else:
            chars.append(r)
    return "".join(chars)


def word_piece_split(word, vocab):
    chars = list(word)
    tokens = []
    start = 0
    max_subwords = 200
    while start < len(chars):
        if len(tokens) >= max_subwords:
            return [BERT_UNK]
        end = len(chars)
        found = False
        while end > start:
            sub = word[start:end]
            if start > 0:
                sub = "##" + sub
            if sub in vocab:
                tokens.append(vocab[sub])
                start = end
                found = True
                break
            end -= 1
        if not found:
            return [BERT_UNK]
    return tokens


def tokenize(texts, vocab):
    input_ids = np.full((len(texts), MAX_SEQ), BERT_PAD, dtype=np.int64)
    attention_mask = np.zeros((len(texts), MAX_SEQ), dtype=np.int64)
    for i, text in enumerate(texts):
        norm = normalize_text(text)
        words = norm.split()
        pos = 1
        input_ids[i, 0] = BERT_CLS
        attention_mask[i, 0] = 1
        for w in words:
            if pos >= MAX_SEQ - 1:
                break
            for tok_id in word_piece_split(w, vocab):
                if pos >= MAX_SEQ - 1:
                    break
                input_ids[i, pos] = tok_id
                attention_mask[i, pos] = 1
                pos += 1
        input_ids[i, pos] = BERT_SEP
        attention_mask[i, pos] = 1
    return input_ids, attention_mask


def mean_pooling(last_hidden, attention_mask):
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    sum_h = np.sum(last_hidden * mask, axis=1)
    cnt = np.maximum(np.sum(mask, axis=1), 1e-9)
    return sum_h / cnt


def save_fbin(path, vecs):
    n, dim = vecs.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(struct.pack("<I", dim))
        # Write in native byte order (little-endian on darwin arm64), matching loadFbin
        vecs.astype("<f4").tofile(f)


def main():
    model_path = os.environ.get("PLASMOD_EMBEDDER_MODEL_PATH", MODEL_PATH)
    vocab_path = os.environ.get("PLASMOD_ONNX_VOCAB_PATH", VOCAB_PATH)

    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(vocab_path):
        print(f"ERROR: vocab not found: {vocab_path}", file=sys.stderr)
        sys.exit(1)

    vocab = load_vocab(vocab_path)
    print(f"[embed] Loaded vocab with {len(vocab)} tokens")

    print(f"[embed] Loading ONNX model: {model_path}")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    out_name = "last_hidden_state" if "last_hidden_state" in output_names else output_names[0]
    print(f"[embed] Inputs: {input_names}, Output: {out_name}")

    def embed(texts, batch_size=64):
        n = len(texts)
        results = np.zeros((n, 384), dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ids, mask = tokenize(texts[start:end], vocab)
            feed = {input_names[0]: ids, input_names[1]: mask}
            if len(input_names) > 2:
                feed[input_names[2]] = np.zeros_like(ids)
            out = sess.run([out_name], feed)[0]
            pooled = mean_pooling(out, mask)
            # L2-normalize
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(norms, 1e-12)
            results[start:end] = pooled.astype(np.float32)
            if start % 500 == 0 or end == n:
                print(f"[embed] {end}/{n}")
        return results

    # Corpus
    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
    out_path = os.path.join(OUTPUT_DIR, "corpus_embedded.fbin")
    if os.path.exists(corpus_path):
        texts = load_jsonl(corpus_path)
        print(f"[embed] Embedding corpus: {len(texts)} texts ...")
        vecs = embed(texts)
        save_fbin(out_path, vecs)
        print(f"[embed] Wrote {out_path} ({vecs.shape})")

    # Queries
    q_path = os.path.join(DATA_DIR, "queries.jsonl")
    q_out = os.path.join(OUTPUT_DIR, "queries_embedded.fbin")
    if os.path.exists(q_path):
        texts = load_jsonl(q_path)
        print(f"[embed] Embedding queries: {len(texts)} texts ...")
        vecs = embed(texts)
        save_fbin(q_out, vecs)
        print(f"[embed] Wrote {q_out} ({vecs.shape})")


if __name__ == "__main__":
    main()
