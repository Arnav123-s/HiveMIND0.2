"""Verify EVERY file of the Hugging Face store against what R1 verified at extract time (manifests/r1-files.json).

    HF_TOKEN=... python3 scripts/verify_store.py --repo asrinna118/hivemind-cells [--out store-verify.json] [--download-all]

Why: R1's upload read back one file per commit (6 of 86). A listing copied by hand from the store had six hashes that
could not be read (one character off, or not 64 hex characters), so this reads the listing from the Hugging Face API
itself, with no copying, and settles each file:

    LFS files (.gguf, .safetensors, rows.jsonl)  the LFS oid the API reports IS the SHA-256 of the bytes: compared with
                                                 the extract-time SHA-256 (--download-all: also downloaded and hashed
                                                 here, so the proof does not rest on the API's oid)
    small files (.json sidecars)                 the API gives a git blob id, not a SHA-256: each is downloaded and
                                                 its SHA-256 compared
    the two rebuilt adapters                     no extract-time hash exists (they were made in R1, not extracted):
                                                 they are rebuilt here from the store's own .gguf, whose hash is
                                                 proven, with training/gguf_to_peft.py, and the rebuild's SHA-256 is
                                                 compared with the stored file's. The rebuild is byte-deterministic
                                                 for the same code and safetensors version (tests/test_resume_tools.py)
    .gitattributes                               Hugging Face's own file: listed, not compared

Exit 0 only when every file the record names is proven and nothing unexpected is stored.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REBUILT = {"g39/t80.g39code.safetensors": "g39/t80.g39code.gguf", "g44/t80.g44words2.safetensors": "g44/t80.g44words2.gguf"}
HF_OWN = {".gitattributes"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected():
    from training import hf_store
    spec = json.loads((ROOT / "manifests" / "r1-files.json").read_text())
    return {hf_store.path_in_store(f["store_name"]): f["sha256"] for a in spec["archives"].values() for f in a["files"]}


def listing(api, repo, revision):
    """{path: {"size", "lfs_sha256" or None}} from the API, at REVISION."""
    from huggingface_hub.hf_api import RepoFile
    out = {}
    for e in api.list_repo_tree(repo, repo_type="model", recursive=True, expand=True, revision=revision):
        if isinstance(e, RepoFile):  # folders are skipped
            out[e.path] = {"size": e.size, "lfs_sha256": e.lfs.sha256 if e.lfs is not None else None}  # BlobLfsInfo.sha256: the oid
    return out


def verify(api, repo, download, download_all=False, rebuild=None):
    """(ok, {path: (status, detail)}, revision) for every stored and every expected file.

    ONE revision of the store is resolved at the start, and the listing and every download are made at it, so the
    store changing while the run reads it cannot mix two versions (audit of 2026-11-28). Each file is fetched once.
    Before an adapter is rebuilt, its fetched source .gguf is hashed again and must equal its manifest SHA-256: the
    proof never rests on bytes that were not themselves checked."""
    revision = api.repo_info(repo, repo_type="model").sha
    if not revision:
        raise SystemExit("verify_store: the store's revision could not be resolved: no proof without a pinned revision")
    want, got = expected(), listing(api, repo, revision)
    res = {}
    tmp = Path(tempfile.mkdtemp(prefix="store-verify-"))
    fetched = {}

    def fetch(p):
        if p not in fetched:
            fetched[p] = Path(download(repo_id=repo, filename=p, repo_type="model", local_dir=str(tmp), revision=revision))
        return fetched[p]

    for p, w in sorted(want.items()):
        g = got.get(p)
        if g is None:
            res[p] = ("MISSING", "in the record, not in the store")
            continue
        if g["lfs_sha256"] and not download_all:
            res[p] = ("PROVEN", "LFS oid = extract-time SHA-256") if g["lfs_sha256"] == w else ("MISMATCH", f"LFS oid {g['lfs_sha256']} != {w}")
        else:
            h = sha(fetch(p))
            res[p] = ("PROVEN", "downloaded, SHA-256 = extract-time SHA-256") if h == w else ("MISMATCH", f"downloaded SHA-256 {h} != {w}")
    for p, src in REBUILT.items():
        g = got.get(p)
        if g is None:
            res[p] = ("MISSING", "rebuilt adapter not in the store")
            continue
        if res.get(src, ("",))[0] != "PROVEN":
            res[p] = ("UNPROVEN", f"its source {src} is not proven")
            continue
        source = fetch(src)
        if sha(source) != want[src]:  # the bytes the rebuild reads, checked themselves, whatever proved src above
            res[p] = ("MISMATCH", f"the fetched source {src} is not the manifest's (SHA-256 {sha(source)} != {want[src]})")
            continue
        out = tmp / "rebuild" / Path(p).name
        (rebuild or _rebuild)(source, out)
        stored = g["lfs_sha256"] if (g["lfs_sha256"] and not download_all) else sha(fetch(p))
        h = sha(out)
        res[p] = ("PROVEN", "rebuilt from the proven .gguf: identical bytes") if h == stored else ("MISMATCH", f"rebuild {h} != stored {stored}")
    for p in sorted(set(got) - set(want) - set(REBUILT)):
        res[p] = ("HF_OWN", "Hugging Face's own file") if p in HF_OWN else ("UNEXPECTED", "stored, but in no record")
    ok = all(s in ("PROVEN", "HF_OWN") for s, _ in res.values())
    return ok, res, revision


def _rebuild(gguf, out):
    from safetensors.numpy import save_file
    from training import gguf_to_peft
    tensors, _meta = gguf_to_peft.read(gguf)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))


def main(argv):
    opts = dict(zip([a for a in argv if a != "--download-all"][::2], [a for a in argv if a != "--download-all"][1::2]))
    if "--repo" not in opts:
        print(__doc__)
        return 2
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    ok, res, revision = verify(api, opts["--repo"], lambda **kw: hf_hub_download(token=os.environ.get("HF_TOKEN"), **kw), "--download-all" in argv)
    counts = {}
    for s, _ in res.values():
        counts[s] = counts.get(s, 0) + 1
    rec = {"repo": opts["--repo"], "revision": revision, "verdict": "ALL PROVEN" if ok else "NOT ALL PROVEN", "counts": counts,
           "download_all": "--download-all" in argv, "files": {p: {"status": s, "detail": d} for p, (s, d) in res.items()}}
    Path(opts.get("--out", "store-verify.json")).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("repo", "revision", "verdict", "counts")}, indent=1))
    for p, (s, d) in res.items():
        if s not in ("PROVEN", "HF_OWN"):
            print(f"  {s}: {p}: {d}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
