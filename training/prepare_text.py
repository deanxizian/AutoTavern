"""Precompute public card rule text embeddings, without any benchmark answers."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from transformers import AutoModel, AutoTokenizer
import taverngym as tg

from .common import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/assets/card-text.pt"))
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(2)
    raw = json.loads((tg.REPO_ROOT / "crates/taverngym/data/cards.json").read_text())
    cards = {c["id"]: c for c in raw["cards"]}
    texts = [""]
    for i in range(1, tg.N_CARDS):
        c = cards[tg.card_info(i)[0]]
        texts.append(re.sub(r"<[^>]+>", "", f'{c["name"]}. {c.get("text", "")}'))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, use_safetensors=True).to(args.device).eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            tokens = tokenizer(texts[i:i + 64], padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(args.device)
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            embeddings.append(torch.nn.functional.normalize(pooled, dim=1).cpu())
    table = torch.cat(embeddings)
    table[0].zero_()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": table, "engine_hash": tg.source_hash()}, args.out)
    metadata = {"model": args.model, "revision": model.config._commit_hash,
                "shape": list(table.shape), "engine_hash": tg.source_hash(),
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    atomic_json(args.out.with_suffix(".json"), metadata)
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
