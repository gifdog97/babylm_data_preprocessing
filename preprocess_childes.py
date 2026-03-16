import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# DOCUMENTATION: """https://talkbank.org/0info/manuals/CHAT.pdf"""

# ====== REQUIRED BY SPEC ======
TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")

# ====== CONFIG ======
MEDIA_ROOT = "/work/gk77/k77035/dataset/childes/childes_media"
TRANSCRIPT_ROOT = "/work/gk77/k77035/dataset/childes/childes_transcript"
OUT_TSV = "output/childes_manifest.tsv"

# ====== EXISTING SETTINGS ======
INCLUDE_TAGS = [
    "%act",
    "%flo",
    "%exp",
    "%par",
    "%com",
    "%gpx",
    "%sit",
    "%int",
    "%add",
]

DEBUG = False
INCLUDE_SPKR = (
    False  # ← manifest の text はスピーカ無しにするのが使いやすいので False 推奨
)


def get_record(filename: str):
    """
    元コードの get_record を残しつつ、今回の manifest では主に使わない想定。
    """
    text = ""
    metadata = ""
    participants = {}
    for l in open(filename, encoding="utf-8", errors="ignore"):
        if l.startswith("@Situation"):
            text += l
        if l.startswith("@Participants"):
            participants.update(
                {
                    kv.split()[0].strip(): kv.split()[1]
                    for kv in l.split("\t")[1].split(",")
                }
            )
        elif l.startswith("@"):
            metadata += l
        elif l.startswith("*"):
            text += l
        elif l.startswith("%"):
            if l.split("\t")[0][:-1] in INCLUDE_TAGS:
                text += l
        elif not l.startswith("\t"):
            # 仕様外の行（必要ならログ）
            pass
    return {
        "filename": filename,
        "text": text,
        "metadata": metadata,
        "participants": participants,
    }


def process_text(textstring: str) -> str:
    # remove time stamps like \x15...\x15
    textstring = re.sub(r".*", "", textstring)
    original = textstring

    # NON SPEECH ROWS
    if DEBUG:
        textstring = re.sub(
            r"(\*\w+:\t)([^\n])+\n%(act|exp|par|com|gpx|sit):\t<bef> (.*?)(\n|^)",
            r"\1[\4]\2\5\2",
            textstring,
        )
        textstring = re.sub(
            r"(\*\w+:\t)([^\n])+\n%(act|exp|par|com|gpx|sit):\t<aft> (.*?)(\n|^)",
            r"\1\2[\4]\5\2",
            textstring,
        )
    else:
        textstring = re.sub(
            r"(\*\w+:\t)([^\n])+\n%(act|exp|par|com|gpx|sit):\t<bef> (.*?)(\n|^)",
            r"\1[\4]\2\5",
            textstring,
        )
        textstring = re.sub(
            r"(\*\w+:\t)([^\n])+\n%(act|exp|par|com|gpx|sit):\t<aft> (.*?)(\n|^)",
            r"\1\2[\4]\5",
            textstring,
        )
    textstring = re.sub(
        r"%(act|exp|par|com|gpx|sit):\t(.*?)(\n|^)", r"[\2]\3", textstring
    )
    textstring = re.sub(r"@Situation:\t(.*?)(\n|^)", r"[\1]\2", textstring)

    # CHILDES annotations cleanup
    textstring = re.sub(r"\[\*[^\]]+?\]", "", textstring)
    textstring = re.sub(r"\[:+.*?\]", "", textstring)
    textstring = re.sub(r"&=0[\w:]+", "", textstring)
    textstring = re.sub(r"&=(\S+)", r"[\1]", textstring)
    for _ in range(5):
        textstring = re.sub(r"\[(\S+)[:_](.*?)\]", r"[\1 \2]", textstring)

    textstring = re.sub(r"@[\w:\$\*]+", "", textstring)
    textstring = re.sub(r"&\+[\w]+", "", textstring)
    textstring = re.sub(r"&\-", "", textstring)
    textstring = re.sub(r"&\~", "", textstring)

    textstring = re.sub(r"\(\w+?\)", "'", textstring)
    textstring = re.sub(r"0\w*", "", textstring)

    textstring = re.sub(r"@l", "", textstring)
    textstring = re.sub(r"(\w)\_", r"\1 ", textstring)

    textstring = re.sub(r"[‡„]", "", textstring)
    textstring = re.sub(r"[↑↓]", "", textstring)
    textstring = re.sub(r"[ˈˌ≠](\w)", r"\1", textstring)
    textstring = re.sub(r"(\w)[:\^]", r"\1", textstring)

    textstring = re.sub(r"&\*\w+:\w+", "", textstring)
    textstring = re.sub(r"\[\^.*?\]", "", textstring)
    textstring = re.sub(r"\(\d*\.+\d*\)", "", textstring)
    textstring = re.sub(r"&\{.*&\}", "", textstring)

    textstring = re.sub(r"\+\.\.\.", "...", textstring)
    textstring = re.sub(r"\+\.\.\?", "...?", textstring)
    textstring = re.sub(r"\+!\?", "!?", textstring)
    textstring = re.sub(r"\+/\.", "...", textstring)
    textstring = re.sub(r"\+/\?", "...?", textstring)
    textstring = re.sub(r"\+//\.", "...", textstring)
    textstring = re.sub(r"\+//\?", "...?", textstring)
    textstring = re.sub(r"\+\.", "", textstring)
    textstring = re.sub(r"\+\"/\.", "", textstring)
    textstring = re.sub(r"\+\"\.", "", textstring)

    textstring = re.sub(r"\+\"(.*?)($|\n)", r'"\1"\2', textstring)
    textstring = re.sub(r"\+(\^|,|\+)", "", textstring)

    textstring = re.sub(r"\[=!\s?(.*)\]", r"[\1]", textstring)
    textstring = re.sub(r"\[!+\]", "", textstring)
    textstring = re.sub(r"\[#.*\]", "", textstring)
    textstring = re.sub(r"\[:.*\]", "", textstring)

    textstring = re.sub(r"\[=\s?(.*?)\]", r"[\1]", textstring)
    textstring = re.sub(r"\[=\?.*?\]", "", textstring)
    textstring = re.sub(r"\[%\s?(.*?)\]", r"[\1]", textstring)
    textstring = re.sub(r"\[\?\]", "", textstring)

    textstring = re.sub(r"\[[<>]\d?\]", "", textstring)
    textstring = re.sub(r"\+<", "", textstring)
    textstring = re.sub(r"(<.+?>|\S+) \[/\]", "", textstring)
    textstring = re.sub(r"(<.+?>|\S+) \[//\]", "", textstring)
    textstring = re.sub(r"\[/\]", "", textstring)
    textstring = re.sub(r"\[//\]", "", textstring)
    textstring = re.sub(r"(<(.+?)>|\S+) \[///\]", r"\1 ...", textstring)
    textstring = re.sub(r"(<(.+?)>|\S+) \[/-\]", r"\1 ...", textstring)
    textstring = re.sub(r"(<(.+?)>|\S+) \[/\?\]", r"", textstring)
    textstring = re.sub(r"(<(.+?)>|\S+) \[(e|\+ exc)\]", r"\1 ...", textstring)
    textstring = re.sub(r"\[\^c.*?\]", "", textstring)

    textstring = re.sub(r"\[\*\]", "", textstring)

    textstring = re.sub(r"( \[\+ \w+\])+\s*[\"\s]?($|\n)", r"\2", textstring)
    textstring = re.sub(r"\t(\[\- \w +\] )+", "", textstring)

    textstring = re.sub(r"[<>]", "", textstring)

    # CLEANUP
    textstring = re.sub(r"  +", " ", textstring)
    textstring = re.sub(r"\t +", "\t", textstring)
    textstring = re.sub(r" +([\.,\?!])", r"\1", textstring)
    textstring = re.sub(r"(^|\n)([\*%]\w+)", r"\1\2:", textstring)

    textstring = re.sub(
        r"(\*\w+:\t)(xxx|yyy|www|0|\.)\s?[\.\?]? ?(\[.*\])\s?[\.\?]? ?($|\n)",
        r"\1\3\4",
        textstring,
    )
    if DEBUG:
        textstring = re.sub(
            r"\*\w+:\t(xxx|yyy|www|0|\.)\s?[\.\?]? ?($|\n)", r"____\2", textstring
        )
    else:
        textstring = re.sub(
            r"\*\w+:\t(xxx|yyy|www|0|\.)\s?[\.\?]? ?($|\n)", "", textstring
        )

    if not INCLUDE_SPKR:
        textstring = re.sub(r"\*\w+:\t", "", textstring)

    lines = textstring.split("\n")
    textstring2 = ""
    l_prev = ""
    for l in lines:
        if l == l_prev:
            continue
        textstring2 += l + "\n"
        l_prev = l

    textstring2 = re.sub(r"\[= :\d+ \]", "", textstring2)

    if DEBUG:
        return "\n".join(
            a + "\n" + b for a, b in zip(textstring2.split("\n"), original.split("\n"))
        )
    else:
        return textstring2


def _ms_to_sec(ms: int) -> float:
    return ms / 1000.0


def _sanitize_utt_id(s: str) -> str:
    # ファイルパス由来IDを安定にする（記号を潰す）
    s = s.replace("\\", "/")
    s = re.sub(r"[^0-9A-Za-z_\-\/\.]", "_", s)
    s = s.replace("/", "_")
    return s


def _extract_speaker(line: str) -> str | None:
    match = re.match(r"^\*(\w+):", line)
    if not match:
        return None
    return match.group(1)


def parse_participants_line(line: str) -> tuple[dict, dict]:
    # Appropriate case (Child / Adult):
    # @Participants:  CHI Target_Child, MOT Mother  PAR0 Participant, PAR1 Participant, PAR2 Participant
    # Inappropriate case (Unknown):
    # @Participants:  PAR0 Participant, PAR1 Participant, PAR2 Participant
    participants_raw_role = {}
    participants_role = {}
    parts = line.split("\t")[1].split(",")
    for part in parts:
        kv = part.strip().split()
        assert len(kv) == 2
        label, participant = kv
        participants_raw_role[label] = participant
        if "Child" in participant:
            participants_role[label] = "Child"
        else:
            participants_role[label] = "Adult"
    if all(participants_role[p] == "Adult" for p in participants_role):
        participants_role = {p: "Unknown" for p in participants_role}
    return participants_raw_role, participants_role


def iter_childes_manifest_rows_for_cha(cha_path: Path, audio_rel_path: str):
    """
    1つの .cha から、'*' 行ごとの manifest 行を生成。
    """
    # audio path は MEDIA_ROOT/rel_path の “相対パス文字列” をそのまま保存（要件: path は音声のパス）
    # ここでは rel_path を保存する（必要なら下で os.path.join(MEDIA_ROOT, rel_path) に変更可）
    audio_path_field = audio_rel_path

    with open(cha_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("@Participants"):
                participants_raw_role, participants_role = parse_participants_line(line)
                continue

            if not line.startswith("*"):
                continue

            speaker = _extract_speaker(line)
            if not speaker:
                continue
            raw_role = participants_raw_role.get(speaker, "Unknown")
            role = participants_role.get(speaker, "Unknown")

            m = TIME_RE.search(line)
            if not m:
                # 時間が無い '*行' はスキップ（要件上 start/end 必須のため）
                continue

            start_ms = int(m.group(1))
            end_ms = int(m.group(2))
            start_sec = _ms_to_sec(start_ms)
            end_sec = _ms_to_sec(end_ms)

            # '*SPK:\tUTT ... \x15...\x15' から発話テキストのみ抽出して正規化
            # process_text は speaker 行形式を想定しているので、1行をそのまま渡す
            cleaned = process_text(line).strip()

            # 空なら捨てる
            if not cleaned:
                continue

            # utt_id を安定生成（rel_path + start/end + 行番号）
            base = _sanitize_utt_id(Path(audio_rel_path).with_suffix("").as_posix())
            utt_id = f"{base}_{start_ms}_{end_ms}_{i}"

            yield {
                "utt_id": utt_id,
                "path": audio_path_field,
                "speaker": speaker,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": cleaned,
                "raw_role": raw_role,
                "role": role,
            }


def main():
    media_root = Path(MEDIA_ROOT)
    transcript_root = Path(TRANSCRIPT_ROOT)
    out_path = Path(OUT_TSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mp3_files = list(media_root.rglob("*.mp3"))

    rows = []
    for mp3_path in tqdm(mp3_files, desc="Scanning mp3"):
        rel_path = mp3_path.relative_to(media_root).as_posix()

        cha_path = transcript_root / Path(rel_path).with_suffix(".cha")
        if not cha_path.exists():
            # 対応する .cha が無い場合はスキップ
            continue

        rows.extend(list(iter_childes_manifest_rows_for_cha(cha_path, rel_path)))

    df = pd.DataFrame(
        rows,
        columns=[
            "utt_id",
            "path",
            "speaker",
            "start_sec",
            "end_sec",
            "text",
            "raw_role",
            "role",
        ],
    )
    df.to_csv(out_path, sep="\t", index=False)
    total_words = df["text"].str.split().str.len().sum()
    total_duration = (df["end_sec"] - df["start_sec"]).sum()
    print(f"Wrote: {out_path}  (n={len(df)})")
    print(f"Total utterances: {len(df)}")
    print(f"Total words: {total_words}")
    print(f"Total duration [sec]: {total_duration:.2f}")
    print(f"Total duration [hour]: {total_duration / 3600:.2f}")


if __name__ == "__main__":
    main()
