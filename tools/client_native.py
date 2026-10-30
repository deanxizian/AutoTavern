"""Isolated native continuation projection; client observations stay authoritative."""
import ctypes
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def library():
    ext = "dylib" if sys.platform == "darwin" else "so"
    path = ROOT / f"runs/client-native/release/libautotavern_client_native.{ext}"
    if not path.is_file():
        raise ValueError("Native continuation bridge is missing; run script/build_client_native.sh")
    lib = ctypes.CDLL(str(path))
    lib.client_project.argtypes, lib.client_project.restype = [ctypes.c_char_p], ctypes.c_void_p
    lib.client_free.argtypes, lib.client_free.restype = [ctypes.c_void_p], None
    import taverngym as tg
    pointer = lib.client_project(b'{"op":"contract"}')
    try:
        contract = json.loads(ctypes.string_at(pointer))["ok"]
    finally:
        lib.client_free(pointer)
    if contract["engine_hash"] != tg.build_info()["source_hash"]:
        raise ValueError("Native continuation engine differs from the loaded training engine")
    return lib

def request(**payload):
    lib = library()
    ptr = lib.client_project(json.dumps(payload, ensure_ascii=False).encode())
    try:
        result = json.loads(ctypes.string_at(ptr))
    finally:
        lib.client_free(ptr)
    if "error" in result:
        raise ValueError(result["error"])
    return result["ok"]

@lru_cache(maxsize=1)
def scripts():
    return request(op="metadata")
