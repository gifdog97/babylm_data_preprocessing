#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from multiprocessing import Pool, get_start_method, shared_memory
from pathlib import Path

import numpy as np
import soundfile as sf

MANIFEST_PATH = Path("./output/childes_manifest.tsv")
CHILDES_ROOT = Path("/work/gk77/k77035/dataset/childes/childes_media")
OUTPUT_DIR = Path("/work/gk77/k77035/dataset/childes/childes_clipped")


def _init_shared(shm_name, shape, dtype, sr_, channels_):
    global SHM, ARR, SHAPE, DTYPE, SR, CHANNELS
    SHM = shared_memory.SharedMemory(name=shm_name)
    ARR = np.ndarray(shape, dtype=dtype, buffer=SHM.buf)
    SHAPE, DTYPE, SR, CHANNELS = shape, dtype, sr_, channels_


def _write_one(args):
    s, e, out = args
    i0 = int(SR * s)
    i1 = int(SR * e)
    # スライスは共有メモリ上のビューなのでコピーは最小
    data = ARR[i0:i1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # PCM16で書く（無圧縮WAV。速くて劣化なし）
    sf.write(out, data, SR, subtype="PCM_16")
    print(f"Wrote: {out} (duration: {e - s:.2f}s, frames: {i1 - i0})")
    return


def parse_segments(manifest_path: Path) -> dict[Path, list[tuple[float, float, str]]]:
    # utt_id  path    speaker start_sec       end_sec text    raw_role        role
    items = defaultdict(list)
    with open(manifest_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out_path, wav_path, s, e = (
                OUTPUT_DIR / f"{row['utt_id']}.wav",
                CHILDES_ROOT / row["path"].replace("mp3", "wav"),
                float(row["start_sec"]),
                float(row["end_sec"]),
            )
            if out_path.exists():
                print(f"File already exists. Skip it.: {out_path}")
                continue
            items[wav_path].append((s, e, out_path))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-j", "--jobs", type=int, default=os.cpu_count(), help="並列ワーカー数"
    )
    args = ap.parse_args()

    segs_dict = parse_segments(MANIFEST_PATH)

    for wav_path, segs in segs_dict.items():
        # 1回だけデコード
        data, sr = sf.read(wav_path, dtype="int16", always_2d=False)
        # [T] or [T, C] を [T] or [T] に揃える（librosa互換系の書き方に近づける）
        if data.ndim == 2:
            channels = data.shape[1]
            # マルチチャネルでもそのまま保存したい場合は reshape しない
            # soundfile は (frames, channels) を受け取るのでこのままでOK
        else:
            channels = 1

        # 共有メモリへ（読み取り専用想定）
        shape = data.shape
        dtype = data.dtype
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        shm_arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        shm_arr[:] = data  # 一回だけコピー

        # sanity: 範囲クリップ
        total_frames = shape[0]

        def clamp(t):
            return max(0.0, min(float(t), total_frames / sr))

        segs = [(clamp(s), clamp(e), out_path) for (s, e, out_path) in segs if e > s]

        # 並列書き出し
        # spawn のほうが共有メモリ初期化が安定（Unixでも）
        if get_start_method(allow_none=True) != "spawn":
            try:
                import multiprocessing as mp

                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

        with Pool(
            processes=args.jobs,
            initializer=_init_shared,
            initargs=(shm.name, shape, dtype, sr, channels),
        ) as pool:
            for _ in pool.imap_unordered(_write_one, segs, chunksize=16):
                pass

        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
