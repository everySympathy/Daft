from __future__ import annotations

import os
import time

import ray
import ray.data
import torch

zero_copy = os.getenv("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "0")
os.environ["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] = zero_copy

ray.init(
    address="local",
    include_dashboard=False,
    runtime_env={"env_vars": {"RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": zero_copy}},
)

shape = (1024, 1024, 256)
tensor = torch.ones(shape, dtype=torch.float32).detach().contiguous()
bytes_one_way = tensor.numel() * tensor.element_size()

ds = ray.data.from_items([{"t": tensor}])


def identity(batch: dict[str, torch.Tensor]):
    assert isinstance(batch["t"], torch.Tensor)
    assert batch["t"].dtype == torch.float32
    return batch


stages = int(os.getenv("N_STAGES", "10"))

for _ in range(stages):
    ds = ds.map_batches(identity)

# ds.materialize()  # warmup

start = time.perf_counter()
ds2 = ds.materialize()
elapsed = time.perf_counter() - start

row = ds2.take(1)[0]
out = row["t"]
assert isinstance(out, torch.Tensor)
assert out.shape == tensor.shape
assert out.dtype == tensor.dtype

total_bytes = bytes_one_way * (stages + 1)
mbps = (total_bytes / (1024 * 1024)) / elapsed

print(f"zero_copy={zero_copy} shape={shape} stages={stages}")
print(f"bytes_one_way: {bytes_one_way}")
print(f"total_bytes(approx): {total_bytes}")
print(f"ray.data(map_batches, torch) elapsed_s={elapsed:.6f} throughput_mib_s={mbps:.2f}")