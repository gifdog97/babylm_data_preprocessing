#!/usr/bin/env python3
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tqdm import tqdm

WAV_ROOT = "/work/gk77/k77035/dataset/bnc/wavs"

# word tierで無視するトークン（必要なら増やしてください）
SKIP_TOKENS = {"sp"}  # sp=短ポーズ
BREAK_TOKENS = {"sp"}  # 発話区切り（ここを調整すると分割の仕方が変わる）


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def parse_word_tier_ooTextFile(path: str) -> List[Interval]:
    """
    Praat TextGrid (ooTextFile) を生テキストで読み、
    IntervalTier "word" の interval を抽出する。
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    n = len(lines)
    intervals: List[Interval] = []

    # 探し方：
    # "IntervalTier"
    # "word"
    # xmin
    # xmax
    # <count>
    # (xmin, xmax, "TEXT") が count 回続く
    while i < n:
        if (
            lines[i].strip() == '"IntervalTier"'
            and i + 1 < n
            and lines[i + 1].strip() == '"word"'
        ):
            # 位置を進めて xmin/xmax/count を読む
            try:
                tier_xmin = float(lines[i + 2].strip())
                tier_xmax = float(lines[i + 3].strip())
                count = int(lines[i + 4].strip())
            except Exception as e:
                raise ValueError(
                    f"Failed to read tier header around line {i} in {path}: {e}"
                )

            j = i + 5
            for k in range(count):
                if j + 2 >= n:
                    break
                try:
                    xmin = float(lines[j].strip())
                    xmax = float(lines[j + 1].strip())
                    txt = _strip_quotes(lines[j + 2])
                except Exception as e:
                    raise ValueError(
                        f"Failed to read interval {k} around line {j} in {path}: {e}"
                    )
                intervals.append(Interval(xmin=xmin, xmax=xmax, text=txt))
                j += 3

            return intervals  # word tierは一つ前提。複数あるなら append 方式に変更。
        i += 1

    raise ValueError(f'No IntervalTier "word" found in: {path}')


def intervals_to_utterances(
    word_intervals: List[Interval],
) -> List[Tuple[float, float, str]]:
    """
    word interval 列から、sp を境に発話（単語列）にまとめる。
    - sp 自体は出力しない
    - {OOV} は単語として残す（不要なら SKIP_TOKENS に追加）
    """
    utts: List[Tuple[float, float, str]] = []
    cur_words: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

    def flush():
        nonlocal cur_words, cur_start, cur_end
        if cur_words and cur_start is not None and cur_end is not None:
            text = " ".join(cur_words)
            utts.append((cur_start, cur_end, text))
        cur_words = []
        cur_start = None
        cur_end = None

    for iv in word_intervals:
        token = iv.text.strip()
        if token in BREAK_TOKENS:
            flush()
            continue
        if token == "":
            continue
        if token in SKIP_TOKENS:
            continue

        if cur_start is None:
            cur_start = iv.xmin
        cur_end = iv.xmax
        cur_words.append(token)

    flush()
    return utts


def truecase_with_moses(text: str, truecase_perl: str, model_path: str) -> str:
    """
    Moses truecase.perl をサブプロセスで呼ぶ（1発話ずつでもOK）。
    """
    import subprocess

    p = subprocess.run(
        ["perl", truecase_perl, "--model", model_path],
        input=text + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    return p.stdout.strip()


def make_manifest(
    textgrid_dir: str,
    out_tsv: str,
    do_truecase: bool = False,
    truecase_perl: str = "truecase.perl",
    truecase_model: str = "truecase.model",
):
    tg_paths = sorted(glob.glob(os.path.join(textgrid_dir, "*.TextGrid")))
    utt_idx = 1

    total_words = 0
    total_duration = 0
    with open(out_tsv, "w", encoding="utf-8") as w:
        w.write("utt_id\tpath\tstart_sec\tend_sec\ttext\n")
        for tg in tqdm(tg_paths):
            base = os.path.basename(tg)
            wav_stem = base.split("_")[0]  # 要件どおり "_" の先頭
            wav_path = os.path.join(WAV_ROOT, wav_stem + ".wav")

            try:
                word_intervals = parse_word_tier_ooTextFile(tg)
                utts = intervals_to_utterances(word_intervals)
            except Exception as e:
                print(f"[WARN] failed to parse: {tg} ({e})")
                continue

            for start, end, txt in utts:
                if do_truecase:
                    # 入力が全大文字前提なら lower してから truecase する流派もあります。
                    # ここはお好みで: txt.lower()
                    txt2 = truecase_with_moses(txt, truecase_perl, truecase_model)
                else:
                    txt2 = txt

                utt_id = f"u{utt_idx:04d}"
                utt_idx += 1
                total_words += len(txt2.split())
                total_duration += end - start
                w.write(f'{utt_id}\t{wav_path}\t{start:.2f}\t{end:.2f}\t"{txt2}"\n')
    print(f"Total utterances: {utt_idx - 1}")
    print(f"Total words: {total_words}")
    print(f"Total duration [sec]: {total_duration:.2f}")
    print(f"Total duration [hour]: {total_duration / 3600:.2f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--textgrid_dir",
        default="/work/gk77/k77035/dataset/bnc/textgrid",
        help="Directory containing .TextGrid files",
    )
    ap.add_argument(
        "-o", "--out", default="output/manifest.tsv", help="Output TSV path"
    )
    ap.add_argument("--truecase", action="store_true", help="Moses truecaser をかける")
    ap.add_argument(
        "--truecase-perl",
        default="truecase.perl",
        help="mosesdecoder/scripts/recaser/truecase.perl のパス",
    )
    ap.add_argument(
        "--truecase-model", default="truecase.model", help="truecase.model のパス"
    )
    args = ap.parse_args()

    make_manifest(
        textgrid_dir=args.textgrid_dir,
        out_tsv=args.out,
        do_truecase=args.truecase,
        truecase_perl=args.truecase_perl,
        truecase_model=args.truecase_model,
    )
