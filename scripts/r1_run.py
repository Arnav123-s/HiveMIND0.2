"""R1 of docs/RESUME-G81.md, in one run: the g81 body out of the Drive archive and into the Hugging Face store, the
two gguf-only cells rebuilt (only if the real-pair validation passes), and the g80 rows audited and quarantined.

    python3 scripts/r1_run.py --drive "/content/drive/MyDrive/HiveMIND recovery — 2026-11-26" \\
                              --work /content/r1 --out "/content/drive/MyDrive/HiveMIND recovery — 2026-11-26/R1-results" \\
                              [--steps restore,extract,validate,audit,upload,report] [--no-upload]

Needs HIVE_STORE=hf, HIVE_HF_REPO and HF_TOKEN for the upload (the token is read by training/hf_store.py from the
environment and never printed). Every step can be run again: what is done and verified is not done twice.

    restore   the 13 archives of manifests/r1-files.json reassembled by the archive's own RESTORE.py (each part's
              and each archive's SHA-256 checked by it), after checking RESTORE.py and RECOVERY-MANIFEST.json
              themselves against the hashes the receipt recorded
    extract   the 84 files taken out of the archives, each checked against its SHA-256 in the archive map, staged
              under the names the store uses (t80.g81code.gguf, ..., rows.jsonl)
    validate  THE GATE (manager ruling on F5): training/gguf_to_peft.py rebuilds t80.g81code and t80.g80words from
              their .gguf and compares them with their real .safetensors. Both must pass (relative error < 2e-3,
              same names and shapes). Only then are g39code and g44words2 rebuilt. If either fails: no rebuild, and
              the report says g82's trunk falls back to 16 leaves, logged.
    audit     scripts/audit_web_rows.py on the g80 rows, with --quarantine: incidence by kind; whether lines
              1068-1074 (the confirmed g34 incident) are flagged; the g38 "walk a site" rows (F10)
    upload    the staged files (and the rebuilt ones) into the store, one commit per generation, then listed and one
              file of each commit downloaded back and its SHA-256 compared
    report    r1-report.json, QUARANTINE.jsonl, web-rows.tsv and the validation reports written to --out
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FILES = ROOT / "manifests" / "r1-files.json"
INCIDENT = range(1068, 1075)  # datasets/g80/rows.jsonl lines 1068-1074 (the verifier seat's audit)
VALIDATE = ("g81code", "g80words")  # two cells that have both files, of two organs
REBUILD = ("g39code", "g44words2")
STEPS = ("restore", "extract", "validate", "audit", "upload", "report")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Run:
    def __init__(self, drive, work, out, upload=True, log=print, files=FILES):
        self.drive, self.work, self.out, self.do_upload, self.log = Path(drive), Path(work), Path(out), upload, log
        self.spec = json.loads(Path(files).read_text())
        self.stage = self.work / "stage"
        self.state_file = self.work / "r1-state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def save(self):
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=1, default=str))

    # --- restore -----------------------------------------------------------------------------------------------
    def restore(self):
        tool, man = self.drive / "RESTORE.py", self.drive / "RECOVERY-MANIFEST.json"
        for f, want in ((tool, self.spec["restore_py_sha256"]), (man, self.spec["recovery_manifest_sha256"])):
            got = sha(f)
            if got != want:
                raise SystemExit(f"r1: {f.name} is not the file the receipt recorded ({got} against {want}): stop")
        out = self.work / "reassembled"
        done = []
        for name, a in self.spec["archives"].items():
            final = out / name
            if final.exists() and sha(final) == a["sha256"]:
                done.append(name)
                continue
            p = subprocess.run([sys.executable, str(tool), "--manifest", str(man), "--parts", str(self.drive), "--output", str(out),
                                "--only", name], capture_output=True, text=True)
            if p.returncode != 0 or not final.exists():
                raise SystemExit(f"r1: RESTORE.py could not reassemble {name}: {(p.stderr or p.stdout).strip()[-400:]}")
            if sha(final) != a["sha256"]:
                raise SystemExit(f"r1: {name}: SHA-256 differs from the manifest after reassembly: stop")
            done.append(name)
            self.log(f"restored {name}")
        self.state["restore"] = {"archives": len(done)}
        self.save()

    # --- extract -----------------------------------------------------------------------------------------------
    def extract(self):
        import zstandard
        self.stage.mkdir(parents=True, exist_ok=True)
        staged = 0
        for name, a in self.spec["archives"].items():
            todo = [f for f in a["files"] if not ((self.stage / f["store_name"]).exists() and sha(self.stage / f["store_name"]) == f["sha256"])]
            if not todo:
                staged += len(a["files"])
                continue
            tar_path = self.work / "tmp" / (name[:-len(".zst")] if name.endswith(".zst") else name + ".tar")
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.work / "reassembled" / name, "rb") as src, open(tar_path, "wb") as dst:
                zstandard.ZstdDecompressor().copy_stream(src, dst)
            try:
                with tarfile.open(tar_path, "r:") as t:
                    for f in todo:
                        try:
                            member = t.getmember(f["path"])
                        except KeyError:
                            member = t.getmember("./" + f["path"])
                        data = t.extractfile(member)  # a hard link resolves to its target's data
                        if data is None:
                            raise SystemExit(f"r1: {f['path']} in {name} is not a file")
                        dest = self.stage / f["store_name"]  # only names from the list are written: no path from the archive
                        with open(dest, "wb") as w:
                            shutil.copyfileobj(data, w)
                        if sha(dest) != f["sha256"]:
                            dest.unlink()
                            raise SystemExit(f"r1: {f['path']} from {name}: SHA-256 differs from the archive map: stop")
            finally:
                tar_path.unlink(missing_ok=True)
            staged += len(a["files"])
            self.log(f"extracted {len(todo)} file(s) from {name}")
        want = sum(len(a["files"]) for a in self.spec["archives"].values())
        if staged != want:
            raise SystemExit(f"r1: {staged} files staged of {want}")
        self.state["extract"] = {"files": staged, "verified": "every file against its archive-map SHA-256"}
        self.save()

    # --- validate (the gate) -----------------------------------------------------------------------------------
    def validate(self):
        from training import gguf_to_peft
        from safetensors.numpy import save_file
        vdir = self.work / "validate"
        vdir.mkdir(parents=True, exist_ok=True)
        reports = {}
        for cell in VALIDATE:
            tensors, meta = gguf_to_peft.read(self.stage / f"t80.{cell}.gguf")
            reports[cell] = gguf_to_peft.compare(tensors, self.stage / f"t80.{cell}.safetensors")
            reports[cell]["config"] = gguf_to_peft.config(tensors, meta)
            self.log(f"validate {cell}: {'PASS' if reports[cell]['ok'] else 'FAIL'} {json.dumps({k: reports[cell].get(k) for k in ('max_rel_error_matrix', 'max_rel_error_edit', 'missing', 'extra')})}")
        passed = all(r["ok"] for r in reports.values())
        rebuilt = []
        if passed:
            for cell in REBUILD:
                tensors, meta = gguf_to_peft.read(self.stage / f"t80.{cell}.gguf")
                cfg = gguf_to_peft.config(tensors, meta)
                ref = reports[VALIDATE[0]]["config"]
                if (cfg["r"], cfg["lora_alpha"]) != (ref["r"], ref["lora_alpha"]):
                    self.log(f"note: {cell} rank/alpha {cfg['r']}/{cfg['lora_alpha']} differ from {VALIDATE[0]}'s {ref['r']}/{ref['lora_alpha']}")
                save_file(tensors, str(self.stage / f"t80.{cell}.safetensors"))
                (vdir / f"t80.{cell}.adapter_config.json").write_text(json.dumps(cfg, indent=1))
                rebuilt.append(cell)
        (vdir / "validation.json").write_text(json.dumps(reports, indent=1, default=str))
        self.state["validate"] = {"gate": "PASS" if passed else "FAIL", "reports": reports, "rebuilt": rebuilt,
                                  "g82_trunk": "18 leaves (g39code, g44words2 rebuilt from .gguf)" if passed else
                                  "FALLBACK: 16 leaves, g39code and g44words2 left out, to be logged in manifests/g82.json"}
        self.save()

    # --- audit -------------------------------------------------------------------------------------------------
    def audit(self):
        import audit_web_rows
        from training import quarantine
        rows_file = self.stage / "rows.jsonl"
        rows, web, flagged, seen, bad = audit_web_rows.audit(rows_file)
        lines = {i for i, _r, _w in flagged}
        have = quarantine.keys(building=True)  # the writer: a missing or empty file is where it starts
        new = 0
        with quarantine.FILE.open("a") as f:
            for i, r, why in sorted(flagged, key=lambda x: x[0]):
                k = quarantine.key(r)
                if k in have:
                    continue
                f.write(json.dumps({"sha256": k, "why": why, "source": "datasets/g80/rows.jsonl", "line": i, "kind": r.get("kind"),
                                    "seed": r.get("seed"), "type": r.get("type")}) + "\n")
                have.add(k)
                new += 1
        tsv = self.work / "web-rows.tsv"
        with tsv.open("w") as f:
            f.write("line\tkind\ttype\tseed\tmember\tsha256\twhy\n")
            for i, r, why in sorted(flagged, key=lambda x: x[0]):
                f.write(f"{i}\t{r.get('kind')}\t{r.get('type')}\t{r.get('seed')}\t{r.get('member')}\t{quarantine.key(r)}\t{why}\n")
        walk = [r for _i, r in rows if r.get("kind") == "walk a site over several pages"]
        self.state["audit"] = {
            "rows": len(rows), "web_rows": len(web), "flagged_rows": len(flagged), "quarantine_keys_added": new,
            "incidence_by_kind": {k: {"answers": seen[k], "flagged": bad[k], "share": round(bad[k] / seen[k], 4) if seen[k] else None}
                                  for k in sorted(set(seen) | set(bad))},
            "incident_lines_1068_1074_flagged": sorted(lines & set(INCIDENT)),
            "incident_confirmed": set(INCIDENT) <= lines,
            "walk_a_site_rows_by_gen": dict(Counter(str(r.get("gen", "?")) for r in walk)),
            "walk_a_site_g38_rows": sum(str(r.get("gen")) == "g38" for r in walk)}
        self.save()

    # --- upload ------------------------------------------------------------------------------------------------
    def upload(self):
        if not self.do_upload:
            self.state["upload"] = {"skipped": "--no-upload"}
            self.save()
            return
        from training import hf_store
        files = sorted(p for p in self.stage.iterdir() if p.is_file())
        groups = {}
        for p in files:
            groups.setdefault(hf_store.path_in_store(p.name).split("/", 1)[0], []).append(p)
        done = self.state.get("upload", {}).get("groups", {})
        for g, ps in sorted(groups.items()):
            if done.get(g, {}).get("verified"):
                continue
            hf_store.put(ps, message=f"R1: {g} from the Drive recovery archive ({len(ps)} files, SHA-256 verified)")
            probe = ps[0]
            back = self.work / "readback" / probe.name
            ok = hf_store.get(probe.name, back) and sha(back) == sha(probe)
            done[g] = {"files": len(ps), "verified": ok, "readback": probe.name}
            if not ok:
                self.state["upload"] = {"groups": done}
                self.save()
                raise SystemExit(f"r1: read-back of {probe.name} from the store did not match: stop")
            self.log(f"uploaded {g}: {len(ps)} files, read-back of {probe.name} matches")
        listed = hf_store.names()
        self.state["upload"] = {"groups": done, "store_files": len(listed), "staged_files": len(files),
                                "all_listed": all(hf_store.path_in_store(p.name) in listed for p in files)}
        self.save()

    # --- report ------------------------------------------------------------------------------------------------
    def report(self):
        from training import quarantine
        self.out.mkdir(parents=True, exist_ok=True)
        self.state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (self.out / "r1-report.json").write_text(json.dumps(self.state, indent=1, default=str))
        for src in (quarantine.FILE, self.work / "web-rows.tsv", self.work / "validate" / "validation.json"):
            if src.exists():
                shutil.copyfile(src, self.out / src.name)
        for p in (self.work / "validate").glob("*.adapter_config.json"):
            shutil.copyfile(p, self.out / p.name)
        v, a, u = self.state.get("validate", {}), self.state.get("audit", {}), self.state.get("upload", {})
        self.log("R1 summary:")
        self.log(f"  files staged and verified: {self.state.get('extract', {}).get('files')}")
        self.log(f"  validation gate: {v.get('gate')}; rebuilt: {v.get('rebuilt')}; g82 trunk: {v.get('g82_trunk')}")
        self.log(f"  audit: {a.get('flagged_rows')} rows flagged of {a.get('web_rows')} web rows; incident 1068-1074 confirmed: "
                 f"{a.get('incident_confirmed')}; quarantine keys added: {a.get('quarantine_keys_added')}")
        self.log(f"  upload: {u}")
        self.log(f"  written to {self.out}")


def main(argv):
    flags = [a for a in argv if a != "--no-upload"]
    opts = dict(zip(flags[::2], flags[1::2]))
    if not {"--drive", "--work", "--out"} <= set(opts):
        print(__doc__)
        return 2
    run = Run(opts["--drive"], opts["--work"], opts["--out"], upload="--no-upload" not in argv)
    steps = opts.get("--steps", ",".join(STEPS)).split(",")
    if run.do_upload and "upload" in steps and (os.environ.get("HIVE_STORE") != "hf" or not os.environ.get("HIVE_HF_REPO") or not os.environ.get("HF_TOKEN")):
        raise SystemExit("r1: the upload needs HIVE_STORE=hf, HIVE_HF_REPO and HF_TOKEN (a notebook secret); or pass --no-upload")
    for s in STEPS:
        if s in steps:
            run.log(f"== {s}")
            getattr(run, s)()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
