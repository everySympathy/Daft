# RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1 python /home/wangzheyan/las-Daft/tests/series/test_tensor_ray_1.py
# RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=0 python /home/wangzheyan/las-Daft/tests/series/test_tensor_ray_1.py

import os
import time

zero_copy = os.getenv("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "0")
os.environ["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] = zero_copy  # 必须在 import ray 前

import ray
import torch

ray.init(runtime_env={"env_vars": {"RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": zero_copy}})

class MyTensorWrapper:
    def __init__(self, tensor):
        self.tensor = tensor

@ray.remote
def bounce(wrapper):
    return wrapper.tensor


# obj = MyTensorWrapper(torch.ones(1024, 1024, 256))
tensor = torch.ones((1024, 1024, 256), dtype=torch.float32).detach().contiguous()
obj = MyTensorWrapper(tensor)

out0 = ray.get(bounce.remote(obj))
assert out0.shape == tensor.shape
assert out0.dtype == tensor.dtype

n = 10
start = time.perf_counter()
for _ in range(n):
    out = ray.get(bounce.remote(obj))
elapsed = time.perf_counter() - start

bytes_one_way = tensor.numel() * tensor.element_size()
total_bytes = bytes_one_way * n * 2
mbps = (total_bytes / (1024 * 1024)) / elapsed

print(f"zero_copy={zero_copy} elapsed={elapsed:.3f}s throughput={mbps:.2f} MiB/s total_bytes={total_bytes}")