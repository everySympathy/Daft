from __future__ import annotations

import os
import time

zero_copy = os.getenv("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "0")
os.environ["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] = zero_copy  # must be set before import ray

import ray
import torch

ray.init(runtime_env={"env_vars": {"RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": zero_copy}})


@ray.remote
class Relay:
    def __init__(self):
        pass

    def identity(self, x):
        return x

"""
RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1 python /home/wangzheyan/las-Daft/tests/series/test_tensor_ray_2.py
有zero-copy-warning
RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=0 python /home/wangzheyan/las-Daft/tests/series/test_tensor_ray_2.py
没有zero-copy-warning,确实慢了
"""
def main() -> None:
    tensor = torch.ones((1024, 1024, 256), dtype=torch.float32).detach().contiguous()
    bytes_one_way = tensor.numel() * tensor.element_size()

    actor_1 = Relay.remote()
    actor_2 = Relay.remote()

    # Warmup: driver -> actor_1 -> actor_2 -> driver
    out0 = ray.get(actor_2.identity.remote(actor_1.identity.remote(tensor)))
    assert out0.shape == tensor.shape
    assert out0.dtype == tensor.dtype

    hops = 100
    start = time.perf_counter()

    ref = actor_1.identity.remote(tensor)
    for i in range(hops):
        ref = (actor_2 if i % 2 == 0 else actor_1).identity.remote(ref)

    out = ray.get(ref)
    elapsed = time.perf_counter() - start

    assert out.shape == tensor.shape
    assert out.dtype == tensor.dtype

    total_bytes = bytes_one_way * (hops + 2)
    mbps = (total_bytes / (1024 * 1024)) / elapsed

    print(f"zero_copy={zero_copy} hops={hops} elapsed={elapsed:.3f}s throughput={mbps:.2f} MiB/s")


if __name__ == "__main__":
    main()