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
def test_serialization(wrapper):
    return wrapper.tensor.sum()

# obj = MyTensorWrapper(torch.ones(1024, 1024, 256))
obj = MyTensorWrapper(torch.ones((1024, 1024, 256), dtype=torch.float32))

# 预热一次避免把 import/actor 启动等开销算进去
ray.get(test_serialization.remote(obj))

n = 10
start = time.perf_counter()
for _ in range(n):
    ray.get(test_serialization.remote(obj))
elapsed = time.perf_counter() - start

total_bytes = obj.tensor.numel() * obj.tensor.element_size() * n
mbps = (total_bytes / (1024 * 1024)) / elapsed

print(f"zero_copy={zero_copy} elapsed={elapsed:.3f}s throughput={mbps:.2f} MiB/s")