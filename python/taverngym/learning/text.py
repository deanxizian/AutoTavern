"""Generate frozen, revision-pinned rule-text vectors aligned to native card indices."""

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

import taverngym as tg

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def card_manifest():
    path = tg.REPO_ROOT / "crates/taverngym/data/cards.json"
    raw = path.read_bytes()
    records = {c["id"]: c for c in json.loads(raw)["cards"]}
    ids = [""] + [tg.card_info(i)[0] for i in range(1, tg.N_CARDS)]
    texts = [""] + [clean_text(records[card].get("text", "")) for card in ids[1:]]
    return ids, texts, hashlib.sha256(raw).hexdigest()


def clean_text(text):
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(text)).split())


def load_text_cache(path):
    cached = torch.load(path, map_location="cpu", weights_only=True)
    ids, _, digest = card_manifest()
    if cached.get("format_version") != 1:
        raise ValueError("Unsupported rule-text cache format")
    if cached["card_data_sha256"] != digest or cached["card_ids"] != ids:
        raise ValueError("Rule-text cache does not match this simulator's card vocabulary")
    values = cached["embeddings"]
    if values.ndim != 2 or values.shape[0] != tg.N_CARDS:
        raise ValueError("Bad rule-text embedding shape")
    if not torch.isfinite(values).all() or torch.count_nonzero(values[0]):
        raise ValueError("Text embeddings must be finite with zero padding at index 0")
    return values.float(), {key: value for key, value in cached.items()
                            if key not in ("embeddings", "card_ids")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/cache/huggingface"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing text cache: {args.output}")
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    ids, texts, digest = card_manifest()
    revision = HfApi().model_info(args.encoder, revision=args.revision).sha
    encoder = SentenceTransformer(
        args.encoder, revision=revision, device=args.device,
        cache_folder=str(args.cache_dir), trust_remote_code=False,
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    unique = sorted(set(text for text in texts if text))
    encoded = encoder.encode(
        unique, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_tensor=True,
    ).detach().float().cpu()
    lookup = {text: i for i, text in enumerate(unique)}
    vectors = torch.zeros(len(ids), encoded.shape[1])
    for index, text in enumerate(texts):
        if text:
            vectors[index] = encoded[lookup[text]]
    metadata = {
        "format_version": 1,
        "encoder": args.encoder,
        "revision": revision,
        "language": "en",
        "preprocessing": "HTML unescape, strip tags, collapse whitespace; rule text only",
        "normalize_embeddings": True,
        "max_seq_length": encoder.max_seq_length,
        "dimension": vectors.shape[1],
        "cards": len(ids),
        "nonempty_texts": sum(bool(text) for text in texts),
        "unique_nonempty_texts": len(unique),
        "card_data_sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_encoder": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(metadata, card_ids=ids, embeddings=vectors), args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
