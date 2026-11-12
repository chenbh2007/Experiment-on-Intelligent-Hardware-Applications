#!/usr/bin/env python3
"""Lab8 孤立数字语音识别。train 可读取带标注的 ZIP 或 record 采集的录音。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Optional, Tuple, Union
from zipfile import BadZipFile, ZipFile
from zipfile import Path as ZipPath

import joblib
import librosa
import numpy as np
import soundfile as sf
from hmmlearn.hmm import GMMHMM
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "digits_wav"
DATA_ZIP = LAB_DIR / "语音数据新.zip"
MODEL_PATH = LAB_DIR / "digit_hmm.pkl"
DIGITS = range(10)
DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
RECORD_RATE = 48000
CHUNK = 1024
FEATURE_RATE = 8000
FFT_SIZE = 256
HOP_LENGTH = 80
N_MFCC = 13
TRIM_DB = 30.0
TARGET_RMS = 0.1  # -20 dBFS
N_STATES = 5
N_MIXTURES = 3
SAMPLE_NAME = re.compile(r"([0-9]+)_([0-9])\.wav", re.IGNORECASE)
GROUPED_SAMPLE_NAME = re.compile(r"([a-z]+)-[0-9]+\.wav", re.IGNORECASE)
AudioPath = Union[Path, ZipPath]
DatasetSplit = Tuple[List[List[AudioPath]], List[Tuple[int, AudioPath]]]


@contextmanager
def open_microphone(rate: int, device: Optional[int], *, manual_stop: bool):
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError("麦克风录音需要 PyAudio，请先安装 PyAudio 和 PortAudio。") from exc

    blocks: List[bytes] = []

    def collect_audio(in_data, _frame_count, _time_info, _status_flags):
        blocks.append(in_data)
        return None, pyaudio.paContinue

    recorder = pyaudio.PyAudio()
    try:
        try:
            stream = recorder.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device,
                frames_per_buffer=CHUNK,
                start=False,
                stream_callback=collect_audio if manual_stop else None,
            )
        except OSError as exc:
            raise RuntimeError(f"无法打开麦克风（{exc}），请检查设备和采样率。") from exc
        try:
            yield stream, blocks
        finally:
            stream.close()
    finally:
        recorder.terminate()


def record_clip(stream, blocks: List[bytes], frame_count: Optional[int], prompt: str) -> np.ndarray:
    while True:
        input(prompt)
        blocks.clear()
        stream.start_stream()
        try:
            if frame_count is None:
                input("录音中，按回车停止。")
            else:
                print("录音中……", flush=True)
                for offset in range(0, frame_count, CHUNK):
                    blocks.append(stream.read(min(CHUNK, frame_count - offset)))
        finally:
            stream.stop_stream()
        audio = np.frombuffer(b"".join(blocks), dtype=np.int16)
        if audio.any():
            return audio
        print("没有录到有效音频，请重新录音。", flush=True)


def recording_frames(duration: Optional[float], rate: int) -> Optional[int]:
    if rate <= 0:
        raise ValueError("录音采样率必须为正。")
    if duration is None:
        return None
    frames = duration * rate
    if not math.isfinite(frames) or frames < 1:
        raise ValueError("录音时长必须为正，且至少能录制一帧。")
    return round(frames)


def load_samples(data_dir: Path) -> List[List[Path]]:
    """按数字分组，保持元数据顺序或按文件名中的录音序号排序。"""
    samples = [[] for _ in DIGITS]
    manifest = data_dir / "meta.json"
    if manifest.is_file():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"{manifest} 应包含一个录音条目列表。")
        seen = set()
        for entry in entries:
            try:
                path = Path(entry["file"])
                digit = int(str(entry["label"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{manifest} 包含无效的 file 或 label 条目。") from exc
            if digit not in DIGITS:
                raise ValueError(f"{manifest} 中的数字标签必须在 0 到 9 之间。")
            if not path.is_absolute():
                # 讲义生成的路径可能包含数据目录名，例如 digits_wav/0_0.wav。
                local_path = data_dir / path
                if not local_path.is_file() and path.parts[:1] == (data_dir.name,):
                    local_path = data_dir.parent / path
                path = local_path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"找不到 {manifest} 中的录音 {path}。")
            if path in seen:
                raise ValueError(f"{manifest} 重复引用了录音 {path}。")
            seen.add(path)
            samples[digit].append(path)
    else:
        indexed = []
        for path in data_dir.iterdir():
            match = SAMPLE_NAME.fullmatch(path.name)
            if match and path.is_file():
                index, digit = map(int, match.groups())
                indexed.append((index, digit, path))
        for _, digit, path in sorted(indexed):
            samples[digit].append(path)
    return samples


def find_grouped_directory(source: AudioPath) -> Optional[AudioPath]:
    for directory in (source, source / "data_good"):
        annotation = directory / "groups.csv"
        # Python 3.8 的 zipfile.Path.is_file() 只检查路径形态。
        if annotation.exists() and annotation.is_file():
            return directory
    return None


def split_grouped_samples(data_dir: AudioPath) -> DatasetSplit:
    """从文件名前缀取标签，使用 groups.csv 的分组避免训练集与测试集共享录音组。"""
    annotation = data_dir / "groups.csv"
    rows = []
    digit_groups = [set() for _ in DIGITS]
    seen = set()
    with StringIO(annotation.read_text(encoding="utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"filename", "group"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{annotation} 必须包含 filename 和 group 两列。")
        for line, row in enumerate(reader, start=2):
            filename = (row["filename"] or "").strip()
            group = (row["group"] or "").strip()
            match = GROUPED_SAMPLE_NAME.fullmatch(filename)
            word = match[1].lower() if match else ""
            if word not in DIGIT_WORDS or not group:
                raise ValueError(f"{annotation} 第 {line} 行的录音文件名或分组无效。")
            if filename in seen:
                raise ValueError(f"{annotation} 重复标注了 {filename}。")
            path = data_dir / filename
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"找不到 {annotation} 中标注的录音 {path}。")
            digit = DIGIT_WORDS.index(word)
            digit_groups[digit].add(group)
            rows.append((digit, path, group))
            seen.add(filename)

    for digit, groups_for_digit in enumerate(digit_groups):
        if len(groups_for_digit) < 2:
            raise ValueError(f"数字 {digit} 至少需要两个不同的录音组，才能分别用于训练和测试。")
    labels = np.array([digit for digit, _, _ in rows])
    groups = [group for _, _, group in rows]
    # 每组整体分配，优先使用五折中的一个留出折，使各类测试样本比例接近 20%。
    folds = min(5, min(map(len, digit_groups)))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    for train_indices, test_indices in splitter.split(labels, labels, groups):
        if len(set(labels[train_indices])) != len(DIGITS) or len(set(labels[test_indices])) != len(DIGITS):
            continue
        training = [[] for _ in DIGITS]
        for index in train_indices:
            digit, path, _ = rows[index]
            training[digit].append(path)
        testing = [(rows[index][0], rows[index][1]) for index in test_indices]
        return training, testing
    raise ValueError(f"无法按 {annotation} 划分出均覆盖 0 到 9 的训练集和测试集。")


def split_dataset(source: AudioPath) -> DatasetSplit:
    grouped_dir = find_grouped_directory(source)
    if grouped_dir is not None:
        return split_grouped_samples(grouped_dir)
    if isinstance(source, ZipPath):
        raise ValueError(f"{source} 中找不到 groups.csv。")
    if not source.is_dir():
        raise FileNotFoundError(f"找不到录音目录 {source}，请先运行 record 或指定 --data-dir / --data-zip。")

    training = []
    testing = []
    for digit, files in enumerate(load_samples(source)):
        if len(files) < 2:
            raise ValueError(f"数字 {digit} 只有 {len(files)} 段录音，每类至少需要 2 段。")
        split = len(files) * 4 // 5
        training.append(files[:split])
        testing.extend((digit, path) for path in files[split:])
    return training, testing


def collect_samples(args: argparse.Namespace) -> None:
    frame_count = recording_frames(args.duration, args.rate)
    if args.per_digit < 1:
        raise ValueError("每类目标录音数必须为正数。")

    data_dir = args.data_dir
    if (data_dir / "meta.json").exists() or find_grouped_directory(data_dir) is not None:
        raise ValueError("该目录已有标注文件，请用 --data-dir 指定本次采集的目录。")
    data_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(data_dir)
    if all(len(files) >= args.per_digit for files in samples):
        print(f"{data_dir} 中每个数字均已达到 {args.per_digit} 段。")
        return

    with open_microphone(args.rate, args.device, manual_stop=frame_count is None) as (stream, blocks):
        for digit, files in enumerate(samples):
            index = int(files[-1].stem.split("_")[0]) if files else 0
            for take in range(len(files) + 1, args.per_digit + 1):
                index += 1
                prompt = f"准备读数字 {digit}（{take}/{args.per_digit}），按回车开始录音。"
                audio = record_clip(stream, blocks, frame_count, prompt)
                path = data_dir / f"{index:03d}_{digit}.wav"
                # 独占创建，避免重复运行时覆盖已有录音。
                output = path.open("xb")
                try:
                    with output:
                        sf.write(output, audio, args.rate, format="WAV", subtype="PCM_16")
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
                print(f"已保存 {path.name}（{len(audio) / args.rate:.2f} 秒）")


def extract_mfcc(
    audio: np.ndarray,
    rate: int,
    *,
    feature_dim: int = N_MFCC,
    trim_db: Optional[float] = TRIM_DB,
    target_rms: Optional[float] = None,
) -> np.ndarray:
    if not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError("音频为空、全静音或含无效采样值。")
    if rate != FEATURE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=FEATURE_RATE)
    if trim_db is not None:
        audio, _ = librosa.effects.trim(
            audio, top_db=trim_db, frame_length=FFT_SIZE, hop_length=HOP_LENGTH
        )
    if target_rms is not None:
        rms = math.sqrt(np.mean(np.square(audio, dtype=np.float64))) if audio.size else 0.0
        if not math.isfinite(rms) or rms == 0.0:
            raise ValueError("音频没有可归一化的能量。")
        audio = np.multiply(audio, target_rms / rms, dtype=np.float64).astype(audio.dtype)
    if len(audio) < FFT_SIZE:
        audio = np.pad(audio, (0, FFT_SIZE - len(audio)))
    mfcc = librosa.feature.mfcc(
        y=audio, sr=FEATURE_RATE, n_mfcc=N_MFCC, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=40
    )
    if feature_dim == 3 * N_MFCC:
        # 短录音使用不超过实际帧数的奇数窗口，至少三帧才能计算二阶差分。
        width = min(9, mfcc.shape[1])
        if width % 2 == 0:
            width -= 1
        delta = librosa.feature.delta(mfcc, width=width, order=1)
        delta2 = librosa.feature.delta(mfcc, width=width, order=2)
        mfcc = np.vstack((mfcc, delta, delta2))
    return mfcc.T.astype(np.float64)


def read_features(
    path: AudioPath,
    *,
    feature_dim: int = N_MFCC,
    trim_db: Optional[float] = TRIM_DB,
    target_rms: Optional[float] = None,
) -> np.ndarray:
    try:
        stream = BytesIO(path.read_bytes()) if isinstance(path, ZipPath) else path.open("rb")
        with stream:
            audio, rate = sf.read(stream, dtype="float32", always_2d=True)
        return extract_mfcc(
            audio.mean(axis=1), rate, feature_dim=feature_dim, trim_db=trim_db, target_rms=target_rms
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"{path} 中的音频无法提取特征（{exc}）") from exc


def score_digit(
    features: np.ndarray, models: List[GMMHMM], scaler: StandardScaler
) -> Tuple[int, np.ndarray]:
    features = scaler.transform(features)
    scores = np.array([model.score(features) / len(features) for model in models])
    if not np.isfinite(scores).all():
        raise ValueError("模型产生了无效得分，请检查训练录音并重新训练。")
    return int(np.argmax(scores)), scores


def create_digit_model(topology: str) -> GMMHMM:
    model = GMMHMM(
        n_components=N_STATES,
        n_mix=N_MIXTURES,
        covariance_type="diag",
        n_iter=20,
        random_state=42,
        covars_prior=1.0,
        covars_weight=1.0,
    )
    if topology == "left-right":
        # 固定起点和转移矩阵的零元素，只学习允许的转移和发射分布。
        model.init_params = "mcw"
        model.params = "tmcw"
        model.startprob_ = np.zeros(N_STATES)
        model.startprob_[0] = 1.0
        model.transmat_ = 0.6 * np.eye(N_STATES) + 0.4 * np.eye(N_STATES, k=1)
        model.transmat_[-1, -1] = 1.0
        # 很短的序列可能到不了末状态，轻微的先验平滑可稳定这些状态的参数。
        model.transmat_prior = 1.01
        model.weights_prior = 1.01
        model.means_weight = 0.01
    return model


def fit_model(
    model_path: Path,
    train_samples: List[List[AudioPath]],
    test_samples: List[Tuple[int, AudioPath]],
    *,
    feature_dim: int,
    topology: str,
    target_rms: Optional[float] = None,
) -> None:
    feature_options = {"feature_dim": feature_dim, "trim_db": TRIM_DB, "target_rms": target_rms}
    print(f"特征 {feature_dim} 维，HMM 拓扑 {topology}，静音裁剪阈值 {TRIM_DB:g} dB", flush=True)
    if target_rms is not None:
        print(f"逐段 RMS 归一化至 {20 * math.log10(target_rms):g} dBFS", flush=True)
    print(f"训练集 {sum(map(len, train_samples))} 段，测试集 {len(test_samples)} 段", flush=True)
    train_features = [[read_features(path, **feature_options) for path in clips] for clips in train_samples]

    # 只用训练集的帧拟合标准化参数，测试和实时识别复用同一个 scaler。
    scaler = StandardScaler().fit(np.vstack([clip for clips in train_features for clip in clips]))
    models = []
    for digit, clips in enumerate(train_features):
        lengths = [len(clip) for clip in clips]
        if sum(lengths) < N_STATES * N_MIXTURES:
            raise ValueError(f"数字 {digit} 的训练语音太短，请重新录制。")
        model = create_digit_model(topology)
        # lengths 保留各段录音的边界，避免把两次发音之间当作状态转移。
        model.fit(scaler.transform(np.vstack(clips)), lengths)
        models.append(model)
        print(f"数字 {digit} 已训练（{len(clips)} 段）", flush=True)

    correct = 0
    for digit, path in test_samples:
        predicted, _ = score_digit(read_features(path, **feature_options), models, scaler)
        correct += predicted == digit
    total = len(test_samples)
    print(f"测试集识别率 {correct}/{total} = {correct / total:.1%}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=model_path.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        joblib.dump(
            {"models": models, "scaler": scaler, "feature_options": feature_options, "topology": topology},
            temporary,
        )
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"模型已保存到 {model_path}")


def train(args: argparse.Namespace) -> None:
    options = {
        "feature_dim": args.features,
        "topology": args.topology,
        "target_rms": TARGET_RMS if args.normalize_rms else None,
    }
    data_zip = args.data_zip
    if data_zip is None and args.data_dir is None and not DATA_DIR.is_dir() and DATA_ZIP.is_file():
        data_zip = DATA_ZIP
    if data_zip is not None:
        with ZipFile(data_zip) as archive:
            fit_model(args.model, *split_dataset(ZipPath(archive)), **options)
    else:
        fit_model(args.model, *split_dataset(args.data_dir or DATA_DIR), **options)


def predict(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"找不到模型 {args.model}，请先运行 train。")
    bundle = joblib.load(args.model)
    # 早期模型使用未经静音裁剪的 13 维 MFCC。
    feature_options = bundle.get("feature_options", {"feature_dim": N_MFCC, "trim_db": None})
    if args.wav is None:
        frame_count = recording_frames(args.duration, args.rate)
        with open_microphone(args.rate, args.device, manual_stop=frame_count is None) as (stream, blocks):
            audio = record_clip(stream, blocks, frame_count, "准备读一个数字，按回车开始录音。")
        features = extract_mfcc(audio.astype(np.float32) / 32768.0, args.rate, **feature_options)
    else:
        features = read_features(args.wav, **feature_options)
    digit, scores = score_digit(features, bundle["models"], bundle["scaler"])
    print(f"识别结果 {digit}")
    print("每帧平均对数似然 " + "  ".join(f"{d}={score:.2f}" for d, score in enumerate(scores)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab8 HMM-GMM 孤立数字语音识别")
    commands = parser.add_subparsers(required=True)

    record_parser = commands.add_parser("record", help="录制 0 到 9 的训练及测试样本")
    record_parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="录音保存目录")
    record_parser.add_argument("--per-digit", type=int, default=10, help="每类目标录音数（默认 10）")
    record_parser.set_defaults(run=collect_samples)

    train_parser = commands.add_parser(
        "train",
        help="划分数据、训练并评估模型",
        description="默认读取 digits_wav 目录，目录缺失时读取同目录下的语音数据新.zip。",
    )
    data_source = train_parser.add_mutually_exclusive_group()
    data_source.add_argument(
        "--data-dir", type=Path, help="包含 groups.csv、meta.json 或 序号_数字.wav 的录音目录"
    )
    data_source.add_argument("--data-zip", type=Path, help="包含 groups.csv 标注的 ZIP 压缩包")
    train_parser.add_argument("--model", type=Path, default=MODEL_PATH, help="模型保存路径")
    train_parser.add_argument(
        "--features",
        type=int,
        choices=(N_MFCC, 3 * N_MFCC),
        default=3 * N_MFCC,
        help="特征维数（默认 39），39 维加入一阶和二阶差分",
    )
    train_parser.add_argument(
        "--topology",
        choices=("ergodic", "left-right"),
        default="ergodic",
        help="HMM 拓扑（默认 ergodic 全连接），left-right 只允许停留或进入下一状态",
    )
    train_parser.add_argument(
        "--normalize-rms", action="store_true", help="裁剪静音后将每段音频的 RMS 归一化至 -20 dBFS"
    )
    train_parser.set_defaults(run=train)

    predict_parser = commands.add_parser("predict", help="识别 WAV 文件或录入一段麦克风语音")
    predict_parser.add_argument("wav", type=Path, nargs="?", help="待识别 WAV 文件，省略时使用麦克风")
    predict_parser.add_argument("--model", type=Path, default=MODEL_PATH, help="本地训练的模型文件")
    predict_parser.set_defaults(run=predict)

    for command in (record_parser, predict_parser):
        command.add_argument("--duration", type=float, help="固定录音秒数，省略时按回车停止")
        command.add_argument("--rate", type=int, default=RECORD_RATE, help="录音采样率（默认 48000）")
        command.add_argument("--device", type=int, help="PyAudio 输入设备编号，省略时使用默认设备")

    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError, RuntimeError, EOFError, BadZipFile, csv.Error) as exc:
        parser.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\n操作已取消。\n")


if __name__ == "__main__":
    main()
