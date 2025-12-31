import zipfile
from pathlib import Path

import browser_cookie3
import requests
import tbdb

BASE_URL = "https://git.talkbank.org/childes/data"
OUT_ROOT = Path("/Volumes/data/childes")
OUT_ROOT.mkdir(exist_ok=True)

cj = browser_cookie3.brave(domain_name="talkbank.org")
# 必要なら 'sla2.talkbank.org' も試す

session = requests.Session()
session.cookies.update(cj)

for subset_name in ["Eng-AAE", "Eng-NA", "Eng-UK"]:
    spec = {
        "corpusName": "childes",
        "corpora": [["childes", subset_name]],
    }

    transcripts = tbdb.getTranscripts(spec)

    col_heads = transcripts["colHeadings"]
    idx_path = col_heads.index("path")  # childes/Eng-AAE/Barriere/fr03
    idx_filename = col_heads.index("filename")

    dataset_name_set = set()
    # extract dataset name
    for row in transcripts["data"]:
        rel_path = Path(row[idx_path]).relative_to(f"childes/{subset_name}")
        dataset_name = rel_path.parts[0]
        if dataset_name in dataset_name_set:
            continue
        dataset_name_set.add(dataset_name)

    for dataset_name in dataset_name_set:
        url = f"{BASE_URL}/{subset_name}/{dataset_name}.zip"

        out_path = OUT_ROOT / subset_name / f"{dataset_name}.zip"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print("GET", url)
        r = session.get(url, allow_redirects=True, stream=True)
        ctype = r.headers.get("Content-Type", "")
        print("  status:", r.status_code, "final_url:", r.url, "ctype:", ctype)

        if "text/html" in ctype:
            head = next(r.iter_content(chunk_size=4000), b"")
            print("skip: got html (login page). head:", head[:200])
            continue

        if r.status_code not in (200, 206):
            print("skip:", r.status_code)
            continue

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        # Extract zip and remove zip file
        extract_dir = out_path.parent
        try:
            with zipfile.ZipFile(out_path, "r") as zf:
                zf.extractall(extract_dir)
            # 正常に解凍できたら zip を削除
            out_path.unlink()
            print(f"  extracted to {extract_dir}, zip removed")
        except zipfile.BadZipFile:
            print(f"  ERROR: bad zip file, keep {out_path}")
