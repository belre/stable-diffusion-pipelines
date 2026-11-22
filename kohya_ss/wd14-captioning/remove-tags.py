import sys, shutil
from pathlib import Path

folder = Path(sys.argv[1])
remove = set(l.strip() for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if l.strip())

bak = folder.parent / (folder.name + "_txt_backup")
bak.mkdir(exist_ok=True)

changed = 0
for t in sorted(folder.glob("*.txt")):
    shutil.copy2(t, bak / t.name)
    tags = [x.strip() for x in t.read_text(encoding="utf-8").split(",") if x.strip()]
    new = [x for x in tags if x not in remove]
    if new != tags:
        changed += 1
    t.write_text(", ".join(new), encoding="utf-8")
print(f"{changed}枚を更新。バックアップ: {bak}")