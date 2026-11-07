"""inspect_latte.py — посмотреть реальную структуру MS-LaTTE.json."""
import json
from pathlib import Path

data = json.loads(Path("MS-LaTTE.json").read_text(encoding="utf-8"))

print(f"top-level type: {type(data).__name__}")
if isinstance(data, dict):
    print(f"keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, list):
            data = v
            print(f"using data[{k!r}] ({len(v)} items)")
            break
        elif isinstance(v, dict) and any(isinstance(x, list) for x in v.values()):
            for k2, v2 in v.items():
                if isinstance(v2, list):
                    data = v2
                    print(f"using data[{k!r}][{k2!r}] ({len(v2)} items)")
                    break
            break

print(f"\nfirst entry type: {type(data[0]).__name__}")
first = data[0]
print(f"first entry keys: {list(first.keys())}")
for k, v in first.items():
    print(f"  {k}: {type(v).__name__} = {repr(v)[:200]}")

# Глубокий разбор Judgements
for field in ("LocJudgements", "TimeJudgements"):
    val = first.get(field)
    print(f"\n{field}: type={type(val).__name__}")
    if isinstance(val, list) and val:
        print(f"  first judgement type={type(val[0]).__name__}")
        if isinstance(val[0], dict):
            print(f"  first judgement keys={list(val[0].keys())}")
            for kk, vv in val[0].items():
                print(f"    {kk}: {type(vv).__name__} = {repr(vv)[:100]}")
    elif isinstance(val, dict):
        print(f"  dict keys={list(val.keys())}")
        for kk, vv in val.items():
            print(f"    {kk}: {type(vv).__name__} = {repr(vv)[:100]}")