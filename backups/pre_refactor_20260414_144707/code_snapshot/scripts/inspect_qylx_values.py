import argparse
from pathlib import Path

import pandas as pd


def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to read CSV with common encodings: {path}") from last_err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--column", default="企业(机构)类型")
    ap.add_argument("--out", default="", help="Optional output TSV path (utf-8-sig)")
    args = ap.parse_args()

    df = read_csv_robust(Path(args.input))
    s = df[args.column].astype(str)
    s = s.str.replace("\u3000", " ", regex=False).str.strip()
    s = s.replace({"": "", "-": "", "—": "", "–": ""})
    s = s[s != ""]

    vc = s.value_counts()
    lines = [f"unique_nonempty={len(vc)}", "value\tcount"]
    lines += [f"{k}\t{int(v)}" for k, v in vc.items()]

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    else:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

