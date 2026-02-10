from __future__ import annotations

import argparse
import io
import os
import pathlib
import tempfile

import daft


def _write_jsonl(path: pathlib.Path, rows: int, payload_bytes: int) -> None:
    filler = "x" * max(payload_bytes, 0)
    with path.open("w", encoding="utf-8", newline="") as f:
        for i in range(rows):
            f.write(f'{{"id":{i},"payload":"{filler}"}}\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--payload-bytes", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024)
    parser.add_argument("--scan-tasks-max-size-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--scan-tasks-min-size-bytes", type=int, default=1)
    args = parser.parse_args()

    if args.path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        base = pathlib.Path(tmp_dir.name)
        jsonl_path = base / "data.jsonl"
        _write_jsonl(jsonl_path, args.rows, args.payload_bytes)
        path = str(jsonl_path)
    else:
        path = args.path

    with daft.context.execution_config_ctx(
        enable_scan_task_split_and_merge=True,
        scan_tasks_min_size_bytes=args.scan_tasks_min_size_bytes,
        scan_tasks_max_size_bytes=args.scan_tasks_max_size_bytes,
    ):
        df = daft.read_json(path, _chunk_size=args.chunk_size)
        s = io.StringIO()
        df.explain(show_all=True, file=s)
        explain_output = s.getvalue()

    os.write(1, explain_output.encode("utf-8"))


if __name__ == "__main__":
    main()
