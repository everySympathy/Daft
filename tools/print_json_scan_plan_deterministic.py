from __future__ import annotations

import argparse
import io
import os
import pathlib
import re
import tempfile

import daft


def _write_fixed_width_jsonl(path: pathlib.Path, *, rows: int, line_bytes: int, id_width: int) -> None:
    prefix = '{"id":"'
    mid = '","payload":"'
    suffix = '"}\n'
    min_bytes = len(prefix) + id_width + len(mid) + len(suffix)
    if line_bytes < min_bytes:
        raise ValueError(f"line_bytes too small: {line_bytes} < {min_bytes}")
    payload_len = line_bytes - min_bytes
    payload = "x" * payload_len
    with path.open("w", encoding="utf-8", newline="") as f:
        for i in range(rows):
            f.write(prefix)
            f.write(str(i).zfill(id_width))
            f.write(mid)
            f.write(payload)
            f.write(suffix)


def _expected_scan_tasks_from_bytes(file_bytes: bytes, split_size_bytes: int) -> int:
    size_bytes = len(file_bytes)
    pos = 0
    offsets = [0]
    while pos < size_bytes:
        target = pos + split_size_bytes
        if target >= size_bytes:
            offsets.append(size_bytes)
            break
        nl = file_bytes.find(b"\n", target)
        end = size_bytes if nl == -1 else nl + 1
        if end <= pos:
            raise RuntimeError(f"split did not advance: pos={pos}, end={end}")
        offsets.append(end)
        pos = end
    return len(offsets) - 1


def _extract_num_scan_tasks(explain_output: str) -> int:
    matches = re.findall(r"\* ScanTaskSource:\s*\n\|\s+Num Scan Tasks = (\d+)", explain_output)
    if not matches:
        raise RuntimeError("Could not find 'ScanTaskSource: Num Scan Tasks = ...' in explain output")
    return int(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--line-bytes", type=int, default=128)
    parser.add_argument("--id-width", type=int, default=8)
    parser.add_argument("--disable-merge", action="store_true")
    parser.add_argument("--assert-exact", action="store_true")
    args = parser.parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    jsonl_path = pathlib.Path(tmp_dir.name) / "data.jsonl"
    _write_fixed_width_jsonl(
        jsonl_path,
        rows=args.rows,
        line_bytes=args.line_bytes,
        id_width=args.id_width,
    )
    file_bytes = jsonl_path.read_bytes()

    expected = _expected_scan_tasks_from_bytes(file_bytes, args.chunk_size)

    with daft.context.execution_config_ctx(
        enable_scan_task_split_and_merge=True,
        scan_tasks_min_size_bytes=0,
        scan_tasks_max_size_bytes=(1 << 60),
        max_sources_per_scan_task=(1 if args.disable_merge else None),
    ):
        df = daft.read_json(str(jsonl_path), _chunk_size=args.chunk_size)
        s = io.StringIO()
        df.explain(show_all=True, file=s)
        explain_output = s.getvalue()

    observed = _extract_num_scan_tasks(explain_output)

    os.write(
        1,
        (
            f"file={jsonl_path}\\n"
            f"file_size_bytes={len(file_bytes)}\\n"
            f"chunk_size={args.chunk_size}\\n"
            f"disable_merge={args.disable_merge}\\n"
            f"expected_num_scan_tasks={expected}\\n"
            f"observed_num_scan_tasks={observed}\\n"
            "\\n"
        ).encode(),
    )
    os.write(1, explain_output.encode("utf-8"))

    if args.assert_exact and expected != observed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
