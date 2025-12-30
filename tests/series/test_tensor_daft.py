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

"""
DAFT_ENABLE_PERF_TESTS=1 RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=0 \
pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor_daft.py -s 

# “开启 zero-copy”（这轮一般也不会提升 Daft tensor，正好用来说明差异来自 torch reducer）
DAFT_ENABLE_PERF_TESTS=1 RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1 \
pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor_daft.py -s
"""
def _init_local_ray(runtime_env: dict | None = None):
    import ray

    if ray.is_initialized():
        ray.shutdown()
    return ray.init(address="local", include_dashboard=False, runtime_env=runtime_env)


@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray runner")
def test_daft_tensor_transfer_like_ray_torch_tensor_script() -> None:
    if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
        pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

    import ray

    _init_local_ray()

    zero_copy = os.getenv("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "0")
    print(f"RAY_ENABLE_ZERO_COPY_TORCH_TENSORS(driver): {zero_copy}")

    shape = (1024, 1024, 256)
    dtype = np.float32

    arr = np.ones(shape, dtype=dtype)
    mp = MicroPartition.from_pydict({"t": [arr]})

    @udf(return_dtype=DataType.tensor(DataType.float32(), shape), use_process=False)
    class Identity1:
        def __init__(self):
            pass

        def __call__(self, t):
            return t

    @udf(return_dtype=DataType.tensor(DataType.float32(), shape), use_process=False)
    class Identity2:
        def __init__(self):
            pass

        def __call__(self, t):
            return t

    execution_config = PyDaftExecutionConfig.from_env()
    resource_request = ResourceRequest(num_cpus=1)

    pool_1 = RayRoundRobinActorPool(
        "daft-tensor-ray-equivalent-1",
        1,
        resource_request,
        ExpressionsProjection([Identity1(daft.col("t")).alias("t")]),
        execution_config=execution_config,
    )
    pool_2 = RayRoundRobinActorPool(
        "daft-tensor-ray-equivalent-2",
        1,
        resource_request,
        ExpressionsProjection([Identity2(daft.col("t")).alias("t")]),
        execution_config=execution_config,
    )

    ppm = PartialPartitionMetadata(num_rows=None, size_bytes=None)

    pool_1.setup()
    pool_2.setup()
    try:
        input_ref = ray.put(mp)

        _, warm_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[input_ref])
        ray.get(warm_ref)

        n_iters = 5

        start = time.perf_counter()
        cur_ref = input_ref
        for _ in range(n_iters):
            _, mid_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[cur_ref])
            _, cur_ref = pool_2.submit(partial_metadatas=[ppm], inputs=[mid_ref])
        out = ray.get(cur_ref)
        elapsed = time.perf_counter() - start

        out_arr = out.to_pydict()["t"][0]
        assert isinstance(out_arr, np.ndarray)
        assert out_arr.shape == arr.shape
        assert out_arr.dtype == arr.dtype
        np.testing.assert_allclose(out_arr.sum(), arr.sum())

        bytes_one_way = arr.nbytes
        total_bytes = bytes_one_way * n_iters * 2
        mbps = (total_bytes / (1024 * 1024)) / elapsed

        print(f"shape={shape} dtype={dtype} bytes_one_way={bytes_one_way}")
        print(f"total_bytes: {total_bytes}")
        print(f"elapsed: {elapsed}")
        print(f"daft_tensor_throughput_mib_s: {mbps:.2f}")
    finally:
        pool_1.teardown()
        pool_2.teardown()


"""
DAFT_ENABLE_PERF_TESTS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor_daft.py -k with_sum -s
"""
@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray runner")
def test_daft_tensor_transfer_like_ray_torch_tensor_script_with_sum() -> None:
    if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
        pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

    import ray

    _init_local_ray()

    shape = (1024, 1024, 256)
    arr = np.ones(shape, dtype=np.float32)
    expected_sum = float(np.sum(arr, dtype=np.float32))

    mp = MicroPartition.from_pydict({"t": [arr]})

    @udf(return_dtype=DataType.float64(), use_process=False)
    class SumTensor:
        def __init__(self):
            pass

        def __call__(self, t):
            return [float(np.sum(x, dtype=np.float32)) for x in t.to_pylist()]

    execution_config = PyDaftExecutionConfig.from_env()
    resource_request = ResourceRequest(num_cpus=64)

    pool = RayRoundRobinActorPool(
        "daft-tensor-ray-equivalent-sum",
        1,
        resource_request,
        ExpressionsProjection([SumTensor(daft.col("t")).alias("s")]),
        execution_config=execution_config,
    )

    ppm = PartialPartitionMetadata(num_rows=None, size_bytes=None)

    pool.setup()
    try:
        _, warm_ref = pool.submit(partial_metadatas=[ppm], inputs=[ray.put(mp)])
        warm_out = ray.get(warm_ref).to_pydict()["s"][0]
        assert warm_out == expected_sum

        n_iters = 10
        start = time.perf_counter()
        last = None
        for _ in range(n_iters):
            _, out_ref = pool.submit(partial_metadatas=[ppm], inputs=[ray.put(mp)])
            last = ray.get(out_ref).to_pydict()["s"][0]
        elapsed = time.perf_counter() - start

        assert last == expected_sum

        bytes_one_way = arr.nbytes
        total_bytes = bytes_one_way * n_iters
        mbps = (total_bytes / (1024 * 1024)) / elapsed

        print(f"shape={shape} bytes_one_way={bytes_one_way}")
        print(f"total_bytes: {total_bytes}")
        print(f"elapsed: {elapsed}")
        print(f"daft_tensor_sum_throughput_mib_s: {mbps:.2f}")
    finally:
        pool.teardown()

"""
DAFT_ENABLE_PERF_TESTS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor_daft.py -k pure_transfer -s
"""
@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray runner")
def test_daft_tensor_transfer_like_ray_torch_tensor_script_pure_transfer() -> None:
    if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
        pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

    import ray

    _init_local_ray()

    shape = (1024, 1024, 256)
    arr = np.ones(shape, dtype=np.float32)
    mp = MicroPartition.from_pydict({"t": [arr]})

    @udf(return_dtype=DataType.tensor(DataType.float32(), shape), use_process=False)
    class Identity:
        def __init__(self):
            pass

        def __call__(self, t):
            return t

    execution_config = PyDaftExecutionConfig.from_env()
    resource_request = ResourceRequest(num_cpus=1)

    pool = RayRoundRobinActorPool(
        "daft-tensor-ray-equivalent-transfer",
        1,
        resource_request,
        ExpressionsProjection([Identity(daft.col("t")).alias("t")]),
        execution_config=execution_config,
    )

    ppm = PartialPartitionMetadata(num_rows=None, size_bytes=None)

    pool.setup()
    try:
        _, warm_ref = pool.submit(partial_metadatas=[ppm], inputs=[ray.put(mp)])
        warm_out = ray.get(warm_ref)
        warm_arr = warm_out.to_pydict()["t"][0]
        assert isinstance(warm_arr, np.ndarray)
        assert warm_arr.shape == arr.shape
        assert warm_arr.dtype == arr.dtype

        n = 10
        start = time.perf_counter()
        last = None
        for _ in range(n):
            in_ref = ray.put(mp)
            _, out_ref = pool.submit(partial_metadatas=[ppm], inputs=[in_ref])
            last = ray.get(out_ref)
        elapsed = time.perf_counter() - start

        out_arr = last.to_pydict()["t"][0]
        assert isinstance(out_arr, np.ndarray)
        assert out_arr.shape == arr.shape
        assert out_arr.dtype == arr.dtype

        bytes_one_way = arr.nbytes
        total_bytes = bytes_one_way * n * 2
        mbps = (total_bytes / (1024 * 1024)) / elapsed

        print(f"shape={shape} bytes_one_way={bytes_one_way}")
        print(f"total_bytes: {total_bytes}")
        print(f"elapsed: {elapsed}")
        print(f"daft_tensor_transfer_throughput_mib_s: {mbps:.2f}")
    finally:
        pool.teardown()

"""
DAFT_ENABLE_PERF_TESTS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor_daft.py -k daft_dataframe_actor_udf_to_actor_udf_pure_transfer -s
RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1 DAFT_ENABLE_PERF_TESTS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor_daft.py -k daft_dataframe_actor_udf_to_actor_udf_pure_transfer -s
"""
@pytest.mark.skipif(get_tests_daft_runner_name() != "ray", reason="requires Ray runner")
def test_daft_dataframe_actor_udf_to_actor_udf_pure_transfer() -> None:
    if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
        pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

    _init_local_ray()

    shape = (1024, 1024, 256)
    arr = np.ones(shape, dtype=np.float32)

    @udf(return_dtype=DataType.tensor(DataType.float32(), shape), use_process=False)
    class Identity:
        def __init__(self):
            pass

        def __call__(self, t):
            return t

    Identity = Identity.with_concurrency(1)

    df = daft.from_pydict({"t": [arr]})

    n = 10
    expr = daft.col("t")
    for _ in range(n):
        expr = Identity(expr)

    df2 = df.select(expr.alias("t"))


    start = time.perf_counter()
    out_df = df2.collect()
    elapsed = time.perf_counter() - start

    out_arr = out_df.to_pydict()["t"][0]
    assert isinstance(out_arr, np.ndarray)
    assert out_arr.shape == arr.shape
    assert out_arr.dtype == arr.dtype

    bytes_one_way = arr.nbytes
    total_bytes = bytes_one_way * (n + 1)
    mbps = (total_bytes / (1024 * 1024)) / elapsed

    print(f"shape={shape} bytes_one_way={bytes_one_way}")
    print(f"total_bytes: {total_bytes}")
    print(f"elapsed: {elapsed}")
    print(f"daft_dataframe_actor_udf_transfer_throughput_mib_s: {mbps:.2f}")