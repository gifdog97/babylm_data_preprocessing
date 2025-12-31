import pathlib

import browser_cookie3
import requests
import tbdb  # TBDBpy

subset_name = "Eng-AAE"  # ["Eng-AAE", "Eng-NA", "Eng-UK"]:

BASE_URL = "https://media.talkbank.org/"
OUT_ROOT = pathlib.Path(f"/Volumes/data/childes_media/{subset_name}")
OUT_ROOT.mkdir(exist_ok=True)

# 1. ブラウザから talkbank のクッキーを引っこ抜く
cj = browser_cookie3.brave(domain_name="talkbank.org")

session = requests.Session()
session.cookies.update(cj)

# 2. TBDBpy で transcript 情報を取る
spec = {
    "corpusName": "childes",
    "corpora": [["childes", subset_name]],
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

    # 拡張子が何かはコーパス次第。Eng-UK は普通 mp3 なのでとりあえず .mp3 を仮定
    if media_field == "audio":
        media_rel = f"{rel_path}.mp3"  # 必要なら .wav に変える
    elif media_field == "video":
        media_rel = f"{rel_path}.mp4"  # 必要なら .avi に変える
    else:  # None
        continue
    url = BASE_URL + media_rel + "?f=save"

    # ローカルの保存先パス
    local_rel = media_rel.replace(f"childes/{subset_name}/", "")
    out_path = OUT_ROOT / local_rel
    if out_path.exists():
        print("SKIP exists:", out_path)
        continue
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("GET", url)
    r = session.get(url, stream=True)

    # 3. ちゃんと音声が返ってきているかチェック
    ctype = r.headers.get("Content-Type", "")
    print("  status:", r.status_code, "ctype:", ctype)

    # ログインページなど HTML っぽいものが返ってきている場合はスキップ
    if "text/html" in ctype:
        print("  -> looks like login page or error, skipping")
        continue

    # 4. バイナリで書き込み
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
