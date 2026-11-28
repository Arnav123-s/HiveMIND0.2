"""The hive's weight store on Hugging Face: the same get and put as training/release_store.py, for the new compute.

The old repository kept every cell (gguf, json, safetensors) and the rows in GitHub Releases (`hive-grown`, ...). The
new home has no releases and no Actions: sessions run on Kaggle or Colab, die after at most 12 hours, and must
checkpoint OUT before they end. This module is that checkpoint: a Hugging Face model repository, private by default.

    HIVE_STORE=hf                         turn it on; release_store.get and put then come here (nothing else changes)
    HIVE_HF_REPO=<user>/<repo>            the store, e.g. asrinna118/hivemind-cells (required)
    HF_TOKEN                              a write token, from the notebook's secrets; never printed or written down

    python3 training/hf_store.py get NAME OUT     the file NAME, saved as OUT (exit 1 when the store has none)
    python3 training/hf_store.py put FILE ...      every file in one commit (all of them land, or none)
    python3 training/hf_store.py ls [PREFIX]       the names the store holds

Where a file sits in the store is worked out from its name alone, so a read needs no listing:
    t80.g81code.safetensors  -> g81/t80.g81code.safetensors      (anything that names a generation: under it)
    rows.jsonl               -> shared/rows.jsonl                 (everything else)
"""
import os
import re
import shutil
import sys
from pathlib import Path

REPO_TYPE = "model"
GEN = re.compile(r"(?<![0-9a-z])g(\d{1,6})(?=[a-z]|trunk|\.|$)")


def path_in_store(name):
    """The store path of a file name (a bare name, never a path: a cell's name is all a reader knows)."""
    name = Path(name).name
    m = GEN.search(name)
    return f"g{int(m.group(1))}/{name}" if m else f"shared/{name}"


def repo_id():
    rid = os.environ.get("HIVE_HF_REPO", "").strip()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", rid):
        raise SystemExit("hf_store: set HIVE_HF_REPO to <user>/<repo> (e.g. asrinna118/hivemind-cells)")
    return rid


def _token():
    return os.environ.get("HF_TOKEN") or None


def _api(api=None):
    if api is not None:
        return api
    from huggingface_hub import HfApi
    return HfApi(token=_token())


def get(name, out, repo=None, api=None, download=None):
    """True when the store holds NAME and it was saved as OUT. Any miss (no file, no repo, no network) is False, as
    with release_store.get, so callers that fall back keep falling back."""
    repo = repo or repo_id()
    if download is None:
        from huggingface_hub import hf_hub_download as download
    try:
        got = download(repo_id=repo, filename=path_in_store(name), repo_type=REPO_TYPE, token=_token())
    except Exception as e:  # noqa: BLE001 - EntryNotFound, RepositoryNotFound, HTTP and network errors all mean "not here"
        print(f"hf_store: {name} not fetched from {repo}: {type(e).__name__}", file=sys.stderr)
        Path(out).unlink(missing_ok=True)
        return False
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(got, out)
    return True


def put(files, repo=None, api=None, message=None):
    """Every file into the store in ONE commit, so a session that dies mid-upload leaves the store as it was."""
    files = [Path(f) for f in files]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"hf_store: not files: {', '.join(missing)}")
    if not files:
        return 0
    repo = repo or repo_id()
    api = _api(api)
    from huggingface_hub import CommitOperationAdd
    api.create_repo(repo_id=repo, repo_type=REPO_TYPE, private=True, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=path_in_store(f.name), path_or_fileobj=str(f)) for f in files]
    api.create_commit(repo_id=repo, repo_type=REPO_TYPE, operations=ops,
                      commit_message=message or f"hive: {len(files)} file(s): {', '.join(f.name for f in files[:6])}")
    for f in files:
        print(f"{repo}: {path_in_store(f.name)}")
    return 0


def names(prefix="", repo=None, api=None):
    return sorted(p for p in _api(api).list_repo_files(repo_id=repo or repo_id(), repo_type=REPO_TYPE) if p.startswith(prefix))


def main(argv):
    if argv[:1] == ["get"] and len(argv) == 3:
        return 0 if get(argv[1], argv[2]) else 1
    if argv[:1] == ["put"]:
        return put(argv[1:])
    if argv[:1] == ["ls"]:
        print("\n".join(names(argv[1] if len(argv) > 1 else "")))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
