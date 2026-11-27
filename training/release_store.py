"""The releases that hold the hive's cells: `hive-grown`, then `hive-grown-2`, `hive-grown-3`, ... when one is full.

GitHub keeps at most 1000 files in one release. On 2026-11-25 `hive-grown` reached 1000 and g78's record (run
45521871368) failed on `file_count limited to 1000 assets per release`, so the line stopped. A cell's files now go to
the first release with room (or to the one that already holds that name, so a re-run replaces it in place), and every
read tries the releases in order. Nothing is deleted or moved. rows.jsonl and failures.jsonl stay in `hive-grown`.

    python3 training/release_store.py get NAME OUT     the file NAME from the first release that has it, saved as OUT
    python3 training/release_store.py put FILE ...      each file into the release that holds its name, else the first
                                                        with room (a new release is made when all are full)
Needs GH_TOKEN and GITHUB_REPOSITORY for put."""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

FIRST = "hive-grown"
MOST_RELEASES = 9  # hive-grown, hive-grown-2 .. hive-grown-9: 9000 files
MOST_FILES = 1000  # GitHub's limit per release


def tags():
    return [FIRST] + [f"{FIRST}-{n}" for n in range(2, MOST_RELEASES + 1)]


def urls(name, repo=None):
    """Where a file may be, in the order to try."""
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "Arnav123-s/HiveMIND")
    return [f"https://github.com/{repo}/releases/download/{t}/{name}" for t in tags()]


def get(name, out, repo=None, fetch=urllib.request.urlretrieve):
    """True when NAME was found in one of the releases and saved as OUT."""
    for url in urls(name, repo):
        try:
            fetch(url, out)
            return True
        except OSError:
            continue
    Path(out).unlink(missing_ok=True)
    return False


def _gh(*args, check=True):
    return subprocess.run(["gh", *args], check=check, capture_output=True, text=True)


def _names(repo, tag):
    """The file names of a release, or None when the release does not exist."""
    got = _gh("api", f"repos/{repo}/releases/tags/{tag}", "--jq", ".id", check=False)
    if got.returncode != 0:
        return None
    rid, names, page = got.stdout.strip(), set(), 1
    while True:
        batch = json.loads(_gh("api", f"repos/{repo}/releases/{rid}/assets?per_page=100&page={page}").stdout)
        names |= {a["name"] for a in batch}
        if len(batch) < 100:
            return names
        page += 1


def place(files, held):
    """{tag: [files]}: a file goes where its name already is, else to the first release with room.
    held: {tag: set of names, or None for a release not made yet}, in the order of tags()."""
    out, room = {}, {t: (MOST_FILES - len(n) if n is not None else MOST_FILES) for t, n in held.items()}
    for f in files:
        name = Path(f).name
        where = next((t for t, n in held.items() if n is not None and name in n), None)
        if where is None:
            where = next((t for t in held if room[t] > 0), None)
            if where is None:
                raise SystemExit(f"::error::all {MOST_RELEASES} releases are full ({MOST_RELEASES * MOST_FILES} files)")
            room[where] -= 1
        out.setdefault(where, []).append(f)
    return out


def put(files, repo=None):
    repo = repo or os.environ["GITHUB_REPOSITORY"]
    held = {t: _names(repo, t) for t in tags()}
    for tag, batch in place(files, held).items():
        if held[tag] is None:
            _gh("release", "create", tag, "-R", repo, "--title", f"What the hive grew ({tag})",
                "--notes", f"More cells of the hive's body: {FIRST} reached GitHub's limit of {MOST_FILES} files.")
        _gh("release", "upload", tag, "-R", repo, *batch, "--clobber")
        print(f"{tag}: {', '.join(Path(f).name for f in batch)}")
    return 0


def main(argv):
    if argv[:1] == ["get"] and len(argv) == 3:
        return 0 if get(argv[1], argv[2]) else 1
    if argv[:1] == ["put"]:
        return put(argv[1:]) if argv[1:] else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
