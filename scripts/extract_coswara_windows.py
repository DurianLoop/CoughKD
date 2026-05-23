import shutil
import tarfile
from pathlib import Path

root = Path(r"D:\CoughKD\datasets\Coswara-Data-master")
out = Path(r"D:\CoughKD\coswara_extracted2")
out.mkdir(exist_ok=True)

dates = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("202"))
print(f"date dirs: {len(dates)}", flush=True)

class ChainFile:
    def __init__(self, parts):
        self.parts = parts
        self.index = 0
        self.current = None
        self._open_next()
    def _open_next(self):
        if self.current:
            self.current.close()
        if self.index >= len(self.parts):
            self.current = None
            return
        self.current = self.parts[self.index].open("rb")
        self.index += 1
    def read(self, size=-1):
        if self.current is None:
            return b""
        if size is None or size < 0:
            chunks = []
            while self.current is not None:
                data = self.current.read()
                if data:
                    chunks.append(data)
                self._open_next()
            return b"".join(chunks)
        chunks = []
        remaining = size
        while remaining > 0 and self.current is not None:
            data = self.current.read(remaining)
            if data:
                chunks.append(data)
                remaining -= len(data)
            else:
                self._open_next()
        return b"".join(chunks)
    def close(self):
        if self.current:
            self.current.close()
            self.current = None

def safe_target(base, name):
    target = (base / name).resolve()
    base_resolved = base.resolve()
    if base_resolved not in target.parents and target != base_resolved:
        raise RuntimeError(f"unsafe tar path: {name}")
    return target

for i, date_dir in enumerate(dates, 1):
    target = out / date_dir.name
    marker = target / ".done"
    if marker.exists():
        print(f"[{i}/{len(dates)}] skip {date_dir.name}", flush=True)
        continue
    parts = sorted(date_dir.glob("*.tar.gz.*"))
    if not parts:
        print(f"[{i}/{len(dates)}] no parts {date_dir.name}", flush=True)
        continue
    print(f"[{i}/{len(dates)}] extract {date_dir.name} parts={len(parts)}", flush=True)
    chain = ChainFile(parts)
    count = 0
    try:
        with tarfile.open(fileobj=chain, mode="r|gz") as tf:
            for member in tf:
                dest = safe_target(out, member.name)
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                if member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src, dest.open("wb") as fh:
                        shutil.copyfileobj(src, fh, length=1024 * 1024)
                    count += 1
        target.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"files={count}\n", encoding="utf-8")
        print(f"[{i}/{len(dates)}] done {date_dir.name} files={count}", flush=True)
    finally:
        chain.close()
print("done", flush=True)
