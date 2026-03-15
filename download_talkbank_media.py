from pathlib import Path

import browser_cookie3
import requests
import tbdb  # TBDBpy


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download media files from CHILDES using TBDBpy and browser cookies."
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        default="childes",
        help="Corpus name to download (e.g., childes, phon, homebank)",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default="Eng-UK",
        help="Subset name to download (e.g., Eng-AAE, Eng-NA, Eng-UK)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Volumes/data/childes_media",
        help="Root directory to save downloaded media",
    )
    return parser.parse_args()


args = parse_args()
BASE_URL = "https://media.talkbank.org/"

OUT_ROOT = Path(args.output) / args.subset_name
OUT_ROOT.mkdir(exist_ok=True, parents=True)

# 1. Extract talkbank cookie info from the browser
# NOTE: change browser from `brave` to anything as needed (chrome, firefox, safari, etc.)
cj = browser_cookie3.brave(domain_name="talkbank.org")

session = requests.Session()
session.cookies.update(cj)

# 2. Get transcript information using TBDBpy
spec = {
    "corpusName": args.corpus_name,
    "corpora": [[args.corpus_name, args.subset_name]],
}
transcripts = tbdb.getTranscripts(spec)

col_heads = transcripts["colHeadings"]
idx_path = col_heads.index("path")  # 例: childes/Eng-UK/Barriere/fr03
idx_filename = col_heads.index("filename")  # 例: fr03（拡張子なし or あり）
idx_media = col_heads.index("media")  # 例: audio


for row in transcripts["data"]:
    rel_path = row[idx_path]
    base_name = row[idx_filename]
    media_field = row[idx_media]

    if media_field == "audio":
        media_rel = f"{rel_path}.mp3"  # 必要なら .wav に変える
    elif media_field == "video":
        media_rel = f"{rel_path}.mp4"  # 必要なら .avi に変える
    else:  # None
        continue
    url = BASE_URL + media_rel + "?f=save"

    # ローカルの保存先パス
    local_rel = media_rel.replace(f"{args.corpus_name}/{args.subset_name}/", "")
    out_path = OUT_ROOT / local_rel
    if out_path.exists():
        print("SKIP exists:", out_path)
        continue
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("GET", url)
    r = session.get(url, stream=True)

    # 3. Check response
    ctype = r.headers.get("Content-Type", "")
    print("  status:", r.status_code, "ctype:", ctype)

    # If the response is HTML, it might be a login page or error
    if "text/html" in ctype:
        print("  -> looks like login page or error, skipping")
        continue

    # 4. Write to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
