from __future__ import annotations

import glob
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# DOCUMENTATION: """https://talkbank.org/manuals/CHAT.pdf"""


INCLUDE_TAGS = [
    "%act",  # Action?
    "%flo",  # this always follows `www .`. TODO: Add a special condition for this?
    "%exp",  # Experimenter's commentary?
    "%par",  # Paralinguistic?
    "%com",  # Commentary?
    "%gpx",  # Gesture?
    "%sit",  # Situation?
    "%int",  # Intonation?
    "%add",  # Addressee
]

# タイムスタンプ: 2524888_2529559 のような形式
TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")

# .cha ファイルのルートディレクトリ
CHILDES_ROOT = "/Volumes/data/childes"

# 切り出した音声を書き出すディレクトリ
OUTPUT_FORMAT = "wav"  # "wav" or "flac"

SEGMENT_DIR = Path("/Volumes/data/childes_segments")


def get_record(filename):
    text = ""
    metadata = ""
    participants = {}
    for l in open(filename):
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
            print(l)
    record = {
        "filename": filename,
        "text": text,
        "metadata": metadata,
        "participants": participants,
    }
    return record


TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")  # 例: \x151234_5678\x15


def _probe_audio_info(media_path: Path) -> tuple[int, int]:
    """
    ffprobe で sample_rate と channels を取る
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "default=nw=1:nk=1",
        str(media_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    # 期待: sample_rate\nchannels
    sr = int(out[0])
    ch = int(out[1])
    return sr, ch


def _decode_mp3_to_pcm_int16(media_path: Path, sr: int, ch: int) -> np.ndarray:
    """
    ffmpegで mp3 -> PCM(s16le) に一発デコードして numpy(int16) にする
    shape: (num_samples, ch)
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        str(ch),
        "-ar",
        str(sr),
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch)
    else:
        audio = audio.reshape(-1, 1)
    return audio


def _write_wav_int16(path: Path, audio_int16: np.ndarray, sr: int) -> None:
    """
    PCM int16 を WAV で保存（標準ライブラリwave使用）
    audio_int16 shape: (n, ch)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ch = audio_int16.shape[1]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def _write_flac_int16(path: Path, audio_int16: np.ndarray, sr: int) -> None:
    """
    PCM int16 を FLAC で保存（libsndfile 経由 / python-soundfile）
    audio_int16 shape: (n, ch)
    - ffmpeg を毎回起動しないので桁違いに高速
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # soundfile は (frames, channels) の int16 をそのまま FLAC(PCM_16) で書ける
    sf.write(
        file=str(path), data=audio_int16, samplerate=sr, format="FLAC", subtype="PCM_16"
    )


def extract_segments_from_cha_numpy_fast(
    cha_path: str | Path,
    childes_root: Path = Path(CHILDES_ROOT),
    segment_root: Path = Path(SEGMENT_DIR),
) -> None:
    """
    1つの .cha から、対応する mp3 を「一回だけデコード」して
    NumPyでスライスして FLAC で高速に切り出す。
    """
    cha_path = Path(cha_path)
    cha_path_rel = cha_path.relative_to(childes_root)

    lines = cha_path.read_text(encoding="utf-8").splitlines(True)

    # @Media 行から media_id を取得
    media_id = None
    for line in lines:
        if line.startswith("@Media:"):
            try:
                media_id, _media_type = line.split("\t", 1)[1].split(",", 1)
                media_id = media_id.strip()
            except Exception:
                pass
            break
    if not media_id:
        print(f"[WARN] No @Media line found in {cha_path}")
        return

    # mp3 パス（質問の置換ロジックを踏襲）
    media_path = Path(str(cha_path).replace("childes", "childes_media")).with_suffix(
        ".mp3"
    )
    if not media_path.exists():
        print(f"[WARN] Audio file not found: {media_path}")
        return

    out_dir = segment_root / cha_path_rel.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ここが高速化の肝：mp3 を一回だけデコード
    sr, ch = _probe_audio_info(media_path)
    pcm = _decode_mp3_to_pcm_int16(media_path, sr=sr, ch=ch)

    seg_idx = 0
    for line in lines:
        if not line.startswith("*"):
            continue
        m = TIME_RE.search(line)
        if not m:
            continue

        start_ms, end_ms = map(int, m.groups())
        speaker = line.split(":", 1)[0].lstrip("*")

        # ms -> sample index
        start = int(start_ms * sr / 1000)
        end = int(end_ms * sr / 1000)

        if end <= start or start < 0 or end > len(pcm):
            # たまにタイムスタンプが音声長とズレることがあるので保険
            continue

        seg_idx += 1
        out_file = out_dir / f"{media_id}_{speaker}_{seg_idx:04d}.{OUTPUT_FORMAT}"
        segment = pcm[start:end]
        if OUTPUT_FORMAT.lower() == "wav":
            _write_wav_int16(out_file.with_suffix(".wav"), segment, sr)
        elif OUTPUT_FORMAT.lower() == "flac":
            _write_flac_int16(out_file.with_suffix(".flac"), segment, sr)
        else:
            raise ValueError(f"Unknown OUTPUT_FORMAT={OUTPUT_FORMAT}")

    print(f"[OK] {cha_path}: wrote {seg_idx} segments -> {out_dir}")


DEBUG = False
INCLUDE_SPKR = True


def process_text(textstring):
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

    # GOING THROUGH CHILDES ANNOTATIONS
    ## Some things just need to be done up top
    ### 10.5 errors
    textstring = re.sub(r"\[\*[^\]]+?\]", "", textstring)  # error

    ### Dealing with colons
    #### 10.3 Explanations and alternatives
    textstring = re.sub(
        r"\[:+.*?\]", "", textstring
    )  # replacements transcriptions get deleted
    #### 9.10 Local events
    textstring = re.sub(
        r"&=0[\w:]+", "", textstring
    )  # this removes things like "sneezes" which someone might want, but we'll exclude it
    textstring = re.sub(r"&=(\S+)", r"[\1]", textstring)
    for _ in range(5):
        textstring = re.sub(r"\[(\S+)[:_](.*?)\]", r"[\1 \2]", textstring)

    ## Going through in order
    ### 8.3 Special form markers
    textstring = re.sub(r"@[\w:\$\*]+", "", textstring)

    ### 8.5 Fragments, Fillers, and Nonwords
    textstring = re.sub(r"&\+[\w]+", "", textstring)
    textstring = re.sub(r"&\-", "", textstring)
    textstring = re.sub(r"&\~", "", textstring)

    ### 8.6 Incomplete and Omitted Words
    textstring = re.sub(r"\(\w+?\)", "'", textstring)
    textstring = re.sub(r"0\w*", "", textstring)

    ### 8.8 Standardized Spellings
    textstring = re.sub(r"@l", "", textstring)
    textstring = re.sub(r"(\w)\_", r"\1 ", textstring)

    ### 9.3 Satellite Markers
    textstring = re.sub(r"[‡„]", "", textstring)

    ### 9.8 Tone Direction
    textstring = re.sub(r"[↑↓]", "", textstring)

    ### 9.9 Prosody Within Words
    textstring = re.sub(r"[ˈˌ≠](\w)", r"\1", textstring)
    textstring = re.sub(r"(\w)[:\^]", r"\1", textstring)

    ### 9.10 Local events
    textstring = re.sub(r"&\*\w+:\w+", "", textstring)  # this removes interposed words
    textstring = re.sub(
        r"\[\^.*?\]", "", textstring
    )  # this removes text descriptions of actions (rare)
    textstring = re.sub(r"\(\d*\.+\d*\)", "", textstring)  # this removes pauses
    textstring = re.sub(r"&\{.*&\}", "", textstring)

    ### 9.11 Utterance terminators
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

    ### 9.12 Utterance linkers
    textstring = re.sub(r"\+\"(.*?)($|\n)", r'"\1"\2', textstring)
    textstring = re.sub(r"\+(\^|,|\+)", "", textstring)

    ### 10.2 Paralinguistic & duration scoping
    textstring = re.sub(
        r"\[=!\s?(.*)\]", r"[\1]", textstring
    )  # paralinguistic material is saved
    textstring = re.sub(r"\[!+\]", "", textstring)  # stress gets deleted
    textstring = re.sub(r"\[#.*\]", "", textstring)  # duration gets deleted
    textstring = re.sub(r"\[:.*\]", "", textstring)  # corrections get deleted

    ### 10.3 Explanations and alternatives
    textstring = re.sub(r"\[=\s?(.*?)\]", r"[\1]", textstring)  # explanations get saved
    textstring = re.sub(
        r"\[=\?.*?\]", "", textstring
    )  # alternative transcriptions get deleted
    textstring = re.sub(r"\[%\s?(.*?)\]", r"[\1]", textstring)  # comments get saved
    textstring = re.sub(r"\[\?\]", "", textstring)  # best guess gets deleted

    ### 10.4 Retracing, Overlap, Exclusions, and Clauses
    textstring = re.sub(r"\[[<>]\d?\]", "", textstring)  # overlap gets deleted
    textstring = re.sub(r"\+<", "", textstring)  # lazy overlap gets deleted
    textstring = re.sub(
        r"(<.+?>|\S+) \[/\]", "", textstring
    )  # repetition gets deleted. This might be controversial
    textstring = re.sub(
        r"(<.+?>|\S+) \[//\]", "", textstring
    )  # repetition gets deleted. This might be controversial
    textstring = re.sub(
        r"\[/\]", "", textstring
    )  # If there's no angle brackets, I can't tell what was repeated, so I keep the repetition but delete the annot.
    textstring = re.sub(
        r"\[//\]", "", textstring
    )  # If there's no angle brackets, I can't tell what was repeated, so I keep the repetition but delete the annot.
    textstring = re.sub(
        r"(<(.+?)>|\S+) \[///\]", r"\1 ...", textstring
    )  # reformulation: annotators aren't consistent about how they use this and it's rare, but I'm gonna keep it
    textstring = re.sub(
        r"(<(.+?)>|\S+) \[/-\]", r"\1 ...", textstring
    )  # false start: annotators aren't consistent about how they use this and it's rare, but I'm gonna keep it
    textstring = re.sub(r"(<(.+?)>|\S+) \[/\?\]", r"", textstring)  # unclear retracing
    textstring = re.sub(
        r"(<(.+?)>|\S+) \[(e|\+ exc)\]", r"\1 ...", textstring
    )  # reformulation: annotators aren't consistent about how they use this and it's rare, but I'm gonna keep it
    textstring = re.sub(r"\[\^c.*?\]", "", textstring)  # clause delimiter

    ### 10.5 errors
    textstring = re.sub(r"\[\*\]", "", textstring)  # error

    ### 10.6 Precodes and Postcodes
    textstring = re.sub(
        r"( \[\+ \w+\])+\s*[\"\s]?($|\n)", r"\2", textstring
    )  # postcode
    textstring = re.sub(r"\t(\[\- \w +\] )+", "", textstring)  # precode

    ### Extra delimiters
    textstring = re.sub(r"[<>]", "", textstring)

    # CLEANUP
    ## fix spaces and puncutation
    textstring = re.sub(r"  +", " ", textstring)
    textstring = re.sub(r"\t +", "\t", textstring)
    textstring = re.sub(r" +([\.,\?!])", r"\1", textstring)
    textstring = re.sub(r"(^|\n)([\*%]\w+)", r"\1\2:", textstring)

    ## remove empty lines
    textstring = re.sub(
        r"(\*\w+:\t)(xxx|yyy|www|0|\.)\s?[\.\?]? ?(\[.*\])\s?[\.\?]? ?($|\n)",
        r"\1\3\4",
        textstring,
    )  # some empty utterance followed by an action
    if DEBUG:
        textstring = re.sub(
            r"\*\w+:\t(xxx|yyy|www|0|\.)\s?[\.\?]? ?($|\n)", r"____\2", textstring
        )
    else:
        textstring = re.sub(
            r"\*\w+:\t(xxx|yyy|www|0|\.)\s?[\.\?]? ?($|\n)", "", textstring
        )

    ## remove speaker?
    if not INCLUDE_SPKR:
        textstring = re.sub(r"\*\w+:\t", "", textstring)

    ## remove repeat lines
    lines = textstring.split("\n")
    textstring = ""
    l_prev = ""
    for l in lines:
        if l == l_prev:
            continue
        textstring += l + "\n"
        l_prev = l

    ## random cleanup
    textstring = re.sub(r"\[= :\d+ \]", "", textstring)

    if DEBUG:
        return "\n".join(
            a + "\n" + b for a, b in zip(textstring.split("\n"), original.split("\n"))
        )
    else:
        return textstring


def incorporate_metadata(text, record):
    # for k, v in record["participants"].items():
    #     text = re.sub(r"\*?"+k, v, text)
    header = "= = = " + record["filename"] + " = = ="
    text = header + "\n" + text
    return text


if __name__ == "__main__":
    out = open("tmp/childes.txt", "w")
    lines = []
    for filename in tqdm(list(glob.iglob(f"{CHILDES_ROOT}/**/*.cha", recursive=True))):
        media_dir = Path(filename.replace("childes", "childes_media")).parent
        if not media_dir.exists():
            print(f"SKIP no media dir: {media_dir}")
            continue
        record = get_record(filename)
        text = process_text(record["text"])
        text = incorporate_metadata(text, record)
        out.write(text)

        # 追加: 音声切り出し
        extract_segments_from_cha_numpy_fast(filename)
