from __future__ import annotations

import os
import time

import numpy as np
import pytest

import daft
from daft import udf
from daft.daft import PyDaftExecutionConfig, ResourceRequest
from daft.datatype import DataType
from daft.expressions import ExpressionsProjection
from daft.recordbatch import MicroPartition
from daft.runners.partitioning import PartialPartitionMetadata
from daft.runners.ray_runner import RayRoundRobinActorPool
from tests.conftest import get_tests_daft_runner_name


def _init_local_ray() -> None:
    import ray

    if ray.is_initialized():
        return

    ray.init(address="local", include_dashboard=False)

"""
RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=0 DAFT_ENABLE_PERF_TESTS=1 pytest -q tests/series/test_tensor_transfer_compare_ray_vs_daft.py::test_ray_actor_to_actor_torch_pure_transfer_no_put -s
RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1 DAFT_ENABLE_PERF_TESTS=1 pytest -q tests/series/test_tensor_transfer_compare_ray_vs_daft.py::test_ray_actor_to_actor_torch_pure_transfer_no_put -s
没问题，符合预期
"""
@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray runner")
def test_ray_actor_to_actor_torch_pure_transfer_no_put() -> None:
    if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
        pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

    try:
        import torch
    except Exception:
        pytest.skip("requires torch")

    import ray

    _init_local_ray()

    zero_copy = os.getenv("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "0")
    print(f"RAY_ENABLE_ZERO_COPY_TORCH_TENSORS(driver): {zero_copy}")

    @ray.remote
    class Relay:
        def __init__(self):
            pass

        def identity(self, x):
            return x

    shape = (1024, 1024, 256)
    tensor = torch.ones(shape, dtype=torch.float32)
    bytes_one_way = tensor.numel() * tensor.element_size()

    tensor_ref = ray.put(tensor)

    actor_1 = Relay.remote()
    actor_2 = Relay.remote()

    # warm = ray.get(actor_2.identity.remote(actor_1.identity.remote(tensor)))
    # assert warm.shape == tensor.shape
    # assert warm.dtype == tensor.dtype

    hops = 10
    start = time.perf_counter()
    ref = actor_1.identity.remote(tensor_ref)
    for i in range(hops - 1):
        ref = (actor_2 if i % 2 == 0 else actor_1).identity.remote(ref)
    out = ray.get(ref)
    elapsed = time.perf_counter() - start

    assert out.shape == tensor.shape
    assert out.dtype == tensor.dtype

    total_bytes = bytes_one_way * hops * 2
    mbps = (total_bytes / (1024 * 1024)) / elapsed

    print(f"shape={shape} bytes_one_way={bytes_one_way}")
    print(f"total_bytes: {total_bytes}")
    print(f"elapsed: {elapsed}")
    print(f"ray_actor_torch_transfer_throughput_mib_s: {mbps:.2f}")

"""
RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=0 DAFT_ENABLE_PERF_TESTS=1 pytest -q tests/series/test_tensor_transfer_compare_ray_vs_daft.py::test_ray_actor_to_actor_numpy_pure_transfer_put -s
很快
"""
@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray runner")
def test_ray_actor_to_actor_numpy_pure_transfer_put() -> None:
    if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
        pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

    import ray

    _init_local_ray()

    @ray.remote
    class Relay:
        def __init__(self):
            pass

        def identity(self, x):
            return x

    shape = (1024, 1024, 256)
    arr = np.ones(shape, dtype=np.float32)
    bytes_one_way = arr.nbytes

    actor_1 = Relay.remote()
    actor_2 = Relay.remote()

    arr_ref = ray.put(arr)

    hops = 10
    start = time.perf_counter()
    ref = actor_1.identity.remote(arr_ref)
    for i in range(hops - 1):
        ref = (actor_2 if i % 2 == 0 else actor_1).identity.remote(ref)
    out = ray.get(ref)
    elapsed = time.perf_counter() - start

    assert isinstance(out, np.ndarray)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype

    total_bytes = bytes_one_way * hops * 2
    mbps = (total_bytes / (1024 * 1024)) / elapsed

    print(f"shape={shape} bytes_one_way={bytes_one_way}")
    print(f"total_bytes: {total_bytes}")
    print(f"elapsed: {elapsed}")
    print(f"ray_actor_numpy_transfer_throughput_mib_s: {mbps:.2f}")


# """
# DAFT_ENABLE_PERF_TESTS=1 pytest -q tests/series/test_tensor_transfer_compare_ray_vs_daft.py::test_daft_actor_pool_tensor_pure_transfer_no_put -s
# """
# @pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray runner")
# def test_daft_actor_pool_tensor_pure_transfer_no_put() -> None:
#     if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
#         pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

#     import ray

#     _init_local_ray()

#     shape = (1024, 1024, 256)
#     arr = np.ones(shape, dtype=np.float32)

#     mp = MicroPartition.from_pydict({"t": [arr]})

#     @udf(return_dtype=DataType.tensor(DataType.float32(), shape), use_process=False)
#     class Identity:
#         def __init__(self):
#             pass

#         def __call__(self, t):
#             return t

#     execution_config = PyDaftExecutionConfig.from_env()
#     resource_request = ResourceRequest(num_cpus=1)

#     pool_1 = RayRoundRobinActorPool(
#         "compare-daft-transfer-1",
#         1,
#         resource_request,
#         ExpressionsProjection([Identity(daft.col("t")).alias("t")]),
#         execution_config=execution_config,
#     )
#     pool_2 = RayRoundRobinActorPool(
#         "compare-daft-transfer-2",
#         1,
#         resource_request,
#         ExpressionsProjection([Identity(daft.col("t")).alias("t")]),
#         execution_config=execution_config,
#     )

#     ppm = PartialPartitionMetadata(num_rows=None, size_bytes=None)

#     pool_1.setup()
#     pool_2.setup()
#     try:
#         cur_ref = ray.put(mp)

#         # _, warm_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[cur_ref])
#         # warm_out = ray.get(warm_ref)
#         # warm_arr = warm_out.to_pydict()["t"][0]
#         # assert isinstance(warm_arr, np.ndarray)
#         # assert warm_arr.shape == arr.shape
#         # assert warm_arr.dtype == arr.dtype

#         hops = 10
#         start = time.perf_counter()
#         for i in range(hops):
#             pool = pool_1 if i % 2 == 0 else pool_2
#             _, cur_ref = pool.submit(partial_metadatas=[ppm], inputs=[cur_ref])
#         out = ray.get(cur_ref)
#         elapsed = time.perf_counter() - start

#         out_arr = out.to_pydict()["t"][0]
#         assert isinstance(out_arr, np.ndarray)
#         assert out_arr.shape == arr.shape
#         assert out_arr.dtype == arr.dtype

#         bytes_one_way = arr.nbytes
#         total_bytes = bytes_one_way * hops * 2
#         mbps = (total_bytes / (1024 * 1024)) / elapsed

#         print(f"shape={shape} bytes_one_way={bytes_one_way}")
#         print(f"total_bytes: {total_bytes}")
#         print(f"elapsed: {elapsed}")
#         print(f"daft_actor_pool_tensor_transfer_throughput_mib_s: {mbps:.2f}")
#     finally:
#         pool_1.teardown()
#         pool_2.teardown()


# @pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray runner")
# def test_daft_actor_pool_numpy_python_pure_transfer_no_put() -> None:
#     if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
#         pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

#     import ray

#     from daft.series import Series

#     _init_local_ray()

#     shape = (1024, 1024, 256)
#     arr = np.ones(shape, dtype=np.float32)

#     mp = MicroPartition.from_pydict({"t": Series.from_pylist([arr], name="t", pyobj="force")})

#     @udf(return_dtype=DataType.python(), use_process=False)
#     class IdentityPy:
#         def __init__(self):
#             pass

#         def __call__(self, t):
#             return t

#     execution_config = PyDaftExecutionConfig.from_env()
#     resource_request = ResourceRequest(num_cpus=1)

#     pool_1 = RayRoundRobinActorPool(
#         "compare-daft-numpy-python-transfer-1",
#         1,
#         resource_request,
#         ExpressionsProjection([IdentityPy(daft.col("t")).alias("t")]),
#         execution_config=execution_config,
#     )
#     pool_2 = RayRoundRobinActorPool(
#         "compare-daft-numpy-python-transfer-2",
#         1,
#         resource_request,
#         ExpressionsProjection([IdentityPy(daft.col("t")).alias("t")]),
#         execution_config=execution_config,
#     )

#     ppm = PartialPartitionMetadata(num_rows=None, size_bytes=None)

#     pool_1.setup()
#     pool_2.setup()
#     try:
#         cur_ref = ray.put(mp)

#         hops = 10
#         start = time.perf_counter()
#         for i in range(hops):
#             pool = pool_1 if i % 2 == 0 else pool_2
#             _, cur_ref = pool.submit(partial_metadatas=[ppm], inputs=[cur_ref])
#         out = ray.get(cur_ref)
#         elapsed = time.perf_counter() - start

#         out_val = out.to_pydict()["t"][0]
#         assert isinstance(out_val, np.ndarray)
#         assert out_val.shape == arr.shape
#         assert out_val.dtype == arr.dtype

#         bytes_one_way = arr.nbytes
#         total_bytes = bytes_one_way * hops * 2
#         mbps = (total_bytes / (1024 * 1024)) / elapsed

#         print(f"shape={shape} bytes_one_way={bytes_one_way}")
#         print(f"total_bytes: {total_bytes}")
#         print(f"elapsed: {elapsed}")
#         print(f"daft_actor_pool_numpy_python_transfer_throughput_mib_s: {mbps:.2f}")
#     finally:
#         pool_1.teardown()
#         pool_2.teardown()