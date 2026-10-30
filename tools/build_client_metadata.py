"""Extract client tags/skin parents from an audited local CardDefs file."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def extract(path):
    data = Path(path).read_bytes()
    root = ET.fromstring(data)
    cards = {}
    for entity in root:
        cid = entity.get("CardID")
        if not cid:
            continue
        tags = {tag.get("name") if not (tag.get("name") or "").isdigit() else tag.get("enumID"): int(tag.get("value"))
                for tag in entity.findall("Tag") if tag.get("value") is not None}
        names = {tag.get("name"): tag for tag in entity.findall("Tag")}
        cards[cid] = {"dbf": int(entity.get("ID")), "tags": tags,
            "name_zh": names["CARDNAME"].findtext("zhCN", "") if "CARDNAME" in names else ""}
    return {"build": root.get("build"), "source_sha256": hashlib.sha256(data).hexdigest(), "cards": cards}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    a.destination.parent.mkdir(parents=True, exist_ok=True)
    a.destination.write_text(json.dumps(extract(a.source), ensure_ascii=False, separators=(",", ":"))+"\n")
