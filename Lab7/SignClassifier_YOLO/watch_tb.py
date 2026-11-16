#!/usr/bin/env python3
"""
将 train.cpp 输出的 train_log.tsv / eval_log.tsv 实时同步到 TensorBoard。

用法:
  持续监听: python watch_tb.py --log-dir ./runs/mute_test/logs
  一次性转换: python watch_tb.py --log-dir ./runs/mute_test/logs --once
"""

import argparse
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def read_new_rows(path: Path, offset: int) -> tuple[list[list[str]], int]:
    """只读取到最后一个完整换行符为止，避免未写完的尾部半行被跳过或丢失。"""
    if not path.exists():
        return [], offset

    file_size = path.stat().st_size
    if file_size < offset:
        offset = 0
    if file_size == offset:
        return [], offset

    with path.open("rb") as f:
        f.seek(offset)
        chunk = f.read()

    last_nl = chunk.rfind(b"\n")
    if last_nl == -1:
        return [], offset

    complete_bytes = chunk[: last_nl + 1]
    new_offset = offset + last_nl + 1
    lines = complete_bytes.decode("utf-8", errors="replace").splitlines()

    rows: list[list[str]] = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        cols = line.split("\t")
        # 文件从头读取时跳过表头
        if offset == 0 and idx == 0 and not cols[0].isdigit():
            continue
        rows.append(cols)
    return rows, new_offset


def log_eval_run(writer: SummaryWriter, run_rows: list[list[str]], per_class: bool) -> None:
    """将单次评估快照（同一 step 的一组阈值结果）汇总写入 TensorBoard。"""
    if not run_rows:
        return

    step = int(run_rows[0][1])
    groups: dict[str, list[tuple[str, float, float, float]]] = defaultdict(list)
    for r in run_rows:
        if len(r) < 8:
            continue
        thresh = f"{float(r[2]):.1f}"
        cls_name = r[4]
        recall = float(r[5])
        fp_ratio = float(r[6])
        avg_iou = float(r[7])
        groups[thresh].append((cls_name, recall, fp_ratio, avg_iou))

    thresh_results: dict[str, tuple[float, float, float, float]] = {}
    for thresh, items in sorted(groups.items()):
        n = len(items)
        if n == 0:
            continue
        f1_sum = 0.0
        recall_sum = 0.0
        prec_sum = 0.0
        iou_sum = 0.0
        valid_iou = 0

        for cls_name, r_val, f_val, iou_val in items:
            f1 = (2.0 * r_val / (1.0 + r_val + f_val)) if r_val > 0.0 else 0.0
            prec = (r_val / (r_val + f_val)) if (r_val + f_val) > 0.0 else 0.0
            recall_sum += r_val
            prec_sum += prec
            f1_sum += f1
            if r_val > 0.0:
                iou_sum += iou_val
                valid_iou += 1

            if per_class:
                writer.add_scalar(f"eval_cls_t{thresh}/{cls_name}_f1", f1, step)
                writer.add_scalar(f"eval_cls_t{thresh}/{cls_name}_iou", iou_val, step)

        macro_recall = recall_sum / n
        macro_prec = prec_sum / n
        macro_f1 = f1_sum / n
        mean_iou = (iou_sum / valid_iou) if valid_iou > 0 else 0.0
        fitness = 0.7 * macro_f1 + 0.3 * mean_iou

        writer.add_scalar(f"eval_t{thresh}/macro_recall", macro_recall, step)
        writer.add_scalar(f"eval_t{thresh}/macro_precision", macro_prec, step)
        writer.add_scalar(f"eval_t{thresh}/macro_f1", macro_f1, step)
        writer.add_scalar(f"eval_t{thresh}/mean_tp_iou", mean_iou, step)
        writer.add_scalar(f"eval_t{thresh}/fitness", fitness, step)
        thresh_results[thresh] = (macro_f1, mean_iou, fitness, macro_recall)

    if thresh_results:
        best_thresh, best_vals = max(thresh_results.items(), key=lambda kv: kv[1][2])
        writer.add_scalar("eval_best/macro_f1", best_vals[0], step)
        writer.add_scalar("eval_best/mean_tp_iou", best_vals[1], step)
        writer.add_scalar("eval_best/fitness", best_vals[2], step)
        writer.add_scalar("eval_best/macro_recall", best_vals[3], step)
        writer.add_scalar("eval_best/best_threshold", float(best_thresh), step)


class EvalRunAssembler:
    """按时间顺序将 eval_log 行切分为独立的评估快照，不合并重复 step。"""

    def __init__(self, expected_rows_per_eval: int = 170) -> None:
        self.expected_rows_per_eval = expected_rows_per_eval
        self.current_step: int | None = None
        self.current_rows: list[list[str]] = []
        self.seen_keys: set[tuple[str, str]] = set()

    def push_rows(self, rows: list[list[str]]) -> list[list[list[str]]]:
        completed_runs: list[list[list[str]]] = []
        for r in rows:
            if len(r) < 8:
                continue
            step = int(r[1])
            key = (f"{float(r[2]):.1f}", r[4])

            # 遇到新 step 或同一 step 内出现重复 (threshold, cls)，说明上一次评估快照已结束
            if self.current_step is not None and (step != self.current_step or key in self.seen_keys):
                completed_runs.append(self.current_rows)
                self.current_rows = []
                self.seen_keys.clear()
                self.current_step = None

            if self.current_step is None:
                self.current_step = step

            self.current_rows.append(r)
            self.seen_keys.add(key)

            if self.expected_rows_per_eval > 0 and len(self.current_rows) >= self.expected_rows_per_eval:
                completed_runs.append(self.current_rows)
                self.current_rows = []
                self.seen_keys.clear()
                self.current_step = None

        return completed_runs

    def flush(self) -> list[list[str]]:
        rem = self.current_rows
        self.current_rows = []
        self.seen_keys.clear()
        self.current_step = None
        return rem


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge C++ YOLO TSV logs to TensorBoard.")
    parser.add_argument("--log-dir", required=True, help="Directory containing train_log.tsv and eval_log.tsv")
    parser.add_argument("--tb-dir", default=None, help="TensorBoard output directory (default: <log-dir>/tb)")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    parser.add_argument("--eval-rows", type=int, default=170, help="Rows per full evaluation pass (default: 5*34=170)")
    parser.add_argument("--per-class", action="store_true", help="Also log per-class F1 and IoU curves")
    parser.add_argument("--once", action="store_true", help="Process existing logs once and exit")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    tb_dir = Path(args.tb_dir) if args.tb_dir else (log_dir / "tb")
    tb_dir.mkdir(parents=True, exist_ok=True)

    train_tsv = log_dir / "train_log.tsv"
    eval_tsv = log_dir / "eval_log.tsv"
    writer = SummaryWriter(str(tb_dir))

    running = True

    def stop_handler(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    train_offset = 0
    eval_offset = 0
    assembler = EvalRunAssembler(expected_rows_per_eval=args.eval_rows)

    print(f"[watch_tb] Watching {log_dir} -> TensorBoard dir {tb_dir}", flush=True)
    try:
        while running:
            train_rows, train_offset = read_new_rows(train_tsv, train_offset)
            for r in train_rows:
                if len(r) < 6:
                    continue
                step = int(r[1])
                writer.add_scalar("train/loss_tot", float(r[2]), step)
                writer.add_scalar("train/loss_cls", float(r[3]), step)
                writer.add_scalar("train/loss_box", float(r[4]), step)
                writer.add_scalar("train/loss_ctr", float(r[5]), step)

            eval_rows, eval_offset = read_new_rows(eval_tsv, eval_offset)
            completed_runs = assembler.push_rows(eval_rows)
            for run in completed_runs:
                log_eval_run(writer, run, args.per_class)

            if args.once:
                tail_run = assembler.flush()
                if tail_run:
                    log_eval_run(writer, tail_run, args.per_class)
                writer.flush()
                break

            if train_rows or completed_runs:
                writer.flush()

            time.sleep(args.interval)
    finally:
        tail_run = assembler.flush()
        if tail_run:
            log_eval_run(writer, tail_run, args.per_class)
        writer.flush()
        writer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
