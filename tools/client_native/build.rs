// Same source identity as the training extension. This adapter lives outside
// both fingerprinted src directories so building it cannot change a checkpoint.
use std::{fs, path::{Path, PathBuf}};
fn collect(root: &Path, paths: &mut Vec<PathBuf>) {
    for e in fs::read_dir(root).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() { collect(&p, paths); }
        else if p.extension().is_some_and(|x| x == "rs") { paths.push(p); }
    }
}
fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = dir.parent().unwrap().parent().unwrap().join("crates");
    let mut files = vec![];
    collect(&root.join("taverngym/src"), &mut files);
    collect(&root.join("taverngym-py/src"), &mut files);
    files.push(root.join("taverngym/data/cards.json"));
    files.sort_by_key(|p| p.strip_prefix(&root).unwrap().to_string_lossy().into_owned());
    let mut hash = 0xcbf29ce484222325u64;
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        let mut crc = !0u32;
        for b in fs::read(&file).unwrap().into_iter().filter(|b| *b != b'\r') {
            crc ^= b as u32;
            for _ in 0..8 { crc = (crc >> 1) ^ (0xedb88320 & (!(crc & 1)).wrapping_add(1)); }
        }
        for b in file.strip_prefix(&root).unwrap().to_string_lossy().bytes().chain([0]).chain((!crc).to_le_bytes()) {
            hash = (hash ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    println!("cargo:rustc-env=CLIENT_ENGINE_HASH={hash:016x}");
}
