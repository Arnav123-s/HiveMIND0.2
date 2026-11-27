"""Download a run's artifacts by name pattern, reading the run's whole list (100 a page, every page).

actions/download-artifact@v4 with `pattern` sees only the first 100 artifacts of a run: a trial with the versus arm has
185, and run 45014809204's record found "100 artifact(s)" and downloaded 0 cells. This pages the list itself.

    python3 scripts/fetch_artifacts.py RUN_ID 'cell-*=got' 'gate-*-[0-9]=gate' ...

Each match is unpacked into <dir>/<artifact name>/ (as download-artifact does with a pattern). A pattern that matches
nothing is said and is not an error (the step after it needs what it needs). Needs GH_TOKEN and GITHUB_REPOSITORY."""
import fnmatch
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


TRIES = 3


def api(path, binary=False, tries=TRIES, wait=3.0):
    """One `gh api` call, tried again on a failure: g75's record (run 45387523263) died on one 502 while downloading
    an artifact, and the line stopped with the generation unrecorded."""
    for n in range(tries):
        got = subprocess.run(["gh", "api", path], capture_output=True)
        if got.returncode == 0:
            return got.stdout if binary else json.loads(got.stdout)
        print(f"gh api {path} failed ({got.returncode}), try {n + 1} of {tries}: {got.stderr[-200:]!r}", file=sys.stderr)
        if n + 1 < tries:
            time.sleep(wait * (n + 1))
    raise subprocess.CalledProcessError(got.returncode, got.args, got.stdout, got.stderr)


def listing(run, repo):
    """Every live artifact of the run, page after page until a page comes back short or empty."""
    got, page = [], 1
    while True:
        batch = api(f"repos/{repo}/actions/runs/{run}/artifacts?per_page=100&page={page}")["artifacts"]
        got += [a for a in batch if not a.get("expired")]
        if len(batch) < 100:
            return got
        page += 1


def wanted(artifacts, specs):
    """(artifact, folder) for each artifact a pattern matches, the first pattern that matches it winning."""
    out = []
    for a in artifacts:
        for pat, folder in specs:
            if fnmatch.fnmatchcase(a["name"], pat):
                out.append((a, folder))
                break
    return out


def main(argv):
    run, specs = argv[0], [s.rsplit("=", 1) for s in argv[1:]]
    repo = os.environ["GITHUB_REPOSITORY"]
    arts = listing(run, repo)
    print(f"the run has {len(arts)} artifact(s)")
    for pat, folder in specs:
        if not any(fnmatch.fnmatchcase(a["name"], pat) for a in arts):
            print(f"nothing matches {pat}")
    for a, folder in wanted(arts, specs):
        dest = Path(folder) / a["name"]
        dest.mkdir(parents=True, exist_ok=True)
        try:
            blob = api(f"repos/{repo}/actions/artifacts/{a['id']}/zip", binary=True)
        except subprocess.CalledProcessError:
            print(f"::warning::{a['name']} could not be downloaded after {TRIES} tries; going on without it")
            continue
        zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
        print(f"{a['name']} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
