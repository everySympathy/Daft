from __future__ import annotations

import copy

import numpy as np
import pyarrow as pa
import pytest

from daft.datatype import DataType, get_super_ext_type
from daft.series import Series
from daft.utils import pyarrow_supports_fixed_shape_tensor
from tests.series import ARROW_FLOAT_TYPES, ARROW_INT_TYPES
from tests.utils import ANSI_ESCAPE

ARROW_VERSION = tuple(int(s) for s in pa.__version__.split(".") if s.isnumeric())
DaftExtension = get_super_ext_type()


@pytest.mark.parametrize("dtype", ARROW_INT_TYPES + ARROW_FLOAT_TYPES)
def test_tensor_roundtrip(dtype):
    np_dtype = dtype.to_pandas_dtype()
    data = [
        np.arange(8, dtype=np_dtype).reshape((2, 2, 2)),
        np.arange(8, 32, dtype=np_dtype).reshape((2, 2, 3, 2)),
        None,
    ]
    s = Series.from_pylist(data, pyobj="allow")

    daft_dtype = DataType.tensor(DataType.from_arrow_type(dtype))

    assert s.datatype() == daft_dtype

    # Test pylist roundtrip.
    back_dtype = DataType.python()
    back = s.cast(back_dtype)

    assert back.datatype() == back_dtype

    out = back.to_pylist()
    np.testing.assert_equal(out, data)

    # Test Arrow roundtrip.
    arrow_arr = s.to_arrow()

    assert isinstance(arrow_arr.type, DaftExtension)
    from_arrow = Series.from_arrow(arrow_arr)

    assert from_arrow.datatype() == s.datatype()
    np.testing.assert_equal(from_arrow.to_pylist(), s.to_pylist())

    s_copy = copy.deepcopy(s)
    assert s_copy.datatype() == s.datatype()
    np.testing.assert_equal(s_copy.to_pylist(), s.to_pylist())


@pytest.mark.parametrize("dtype", ARROW_INT_TYPES + ARROW_FLOAT_TYPES)
def test_fixed_shape_tensor_roundtrip(dtype):
    np_dtype = dtype.to_pandas_dtype()
    shape = (3, 2, 2)
    data = [
        np.arange(12, dtype=np_dtype).reshape(shape),
        np.arange(12, 24, dtype=np_dtype).reshape(shape),
        None,
    ]
    s = Series.from_pylist(data, pyobj="allow")

    target_dtype = DataType.tensor(DataType.from_arrow_type(dtype), shape)

    t = s.cast(target_dtype)

    assert t.datatype() == target_dtype

    # Test pylist roundtrip.
    back_dtype = DataType.python()
    back = t.cast(back_dtype)

    assert back.datatype() == back_dtype

    out = back.to_pylist()
    np.testing.assert_equal(out, data)

    # Test Arrow roundtrip.
    arrow_arr = t.to_arrow()

    if pyarrow_supports_fixed_shape_tensor():
        assert arrow_arr.type == pa.fixed_shape_tensor(dtype, shape)
    else:
        assert isinstance(arrow_arr.type, DaftExtension)
    from_arrow = Series.from_arrow(t.to_arrow())

    assert from_arrow.datatype() == t.datatype()
    np.testing.assert_equal(from_arrow.to_pylist(), t.to_pylist())

    if ARROW_VERSION >= (12, 0, 0):
        # Can't deepcopy pyarrow's fixed-shape tensor type.
        t_copy = t
    else:
        t_copy = copy.deepcopy(t)
    assert t_copy.datatype() == t.datatype()
    np.testing.assert_equal(t_copy.to_pylist(), t.to_pylist())


@pytest.mark.parametrize("dtype", ARROW_INT_TYPES + ARROW_FLOAT_TYPES)
@pytest.mark.parametrize("fixed_shape", [True, False])
def test_tensor_numpy_inference(dtype, fixed_shape):
    np_dtype = dtype.to_pandas_dtype()
    if fixed_shape:
        shape = (2, 2)
        arr = np.arange(np.prod(shape), dtype=np_dtype).reshape(shape)
        arrs = [arr, arr, None]
    else:
        shape1 = (2, 2)
        shape2 = (3, 3)
        arr1 = np.arange(np.prod(shape1), dtype=np_dtype).reshape(shape1)
        arr2 = np.arange(np.prod(shape1), np.prod(shape1) + np.prod(shape2), dtype=np_dtype).reshape(shape2)
        arrs = [arr1, arr2, None]
    s = Series.from_pylist(arrs, pyobj="allow")
    assert s.datatype() == DataType.tensor(DataType.from_arrow_type(dtype))
    out = s.to_pylist()
    np.testing.assert_equal(out, arrs)


def test_tensor_repr():
    arr = np.arange(np.prod((2, 2)), dtype=np.int64).reshape((2, 2))
    arrs = [arr, arr, None]
    s = Series.from_pylist(arrs, pyobj="allow")

    out_repr = ANSI_ESCAPE.sub("", repr(s))
    assert (
        out_repr.replace("\r", "")
        == """╭───────────────────────╮
│ list_series           │
│ ---                   │
│ Tensor[Int64]         │
╞═══════════════════════╡
│ <Tensor shape=(2, 2)> │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ <Tensor shape=(2, 2)> │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ None                  │
╰───────────────────────╯
"""
    )


"""
# DAFT_ENABLE_PERF_TESTS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor.py -k test_tensor_object_store_serde_copy_overhead_between_actor_udfs -s

DAFT_ENABLE_PERF_TESTS=1 RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor.py -k test_tensor_object_store_serde_copy_overhead_between_actor_udfs -s
"""
def test_tensor_object_store_serde_copy_overhead_between_actor_udfs() -> None:
    import os
    import time

    from tests.conftest import get_tests_daft_runner_name

    if get_tests_daft_runner_name() != "ray":
        pytest.skip("requires Ray runner")

    if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
        pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

    try:
        import torch
    except Exception:
        pytest.skip("requires torch")

    import ray

    import daft
    from daft import udf
    from daft.daft import PyDaftExecutionConfig, ResourceRequest
    from daft.datatype import DataType
    from daft.expressions import ExpressionsProjection
    from daft.recordbatch import MicroPartition
    from daft.runners.partitioning import PartialPartitionMetadata
    from daft.runners.ray_runner import RayRoundRobinActorPool

    class Wrapper:
        def __init__(self, t):
            self.t = t

    shape = (8, 1024, 1024)
    n_rows = 8
    tensors = [torch.ones(shape, dtype=torch.float32) * float(i) for i in range(n_rows)]
    rows = [Wrapper(t) for t in tensors]

    @udf(return_dtype=DataType.python(), use_process=False)
    class Identity1:
        def __init__(self):
            pass

        def __call__(self, t):
            return t

    @udf(return_dtype=DataType.python(), use_process=False)
    class Identity2:
        def __init__(self):
            pass

        def __call__(self, t):
            return t

    execution_config = PyDaftExecutionConfig.from_env()
    resource_request = ResourceRequest(num_cpus=1)

    pool_1 = RayRoundRobinActorPool(
        "torch-tensor-serde-overhead-1",
        1,
        resource_request,
        ExpressionsProjection([Identity1(daft.col("t")).alias("t")]),
        execution_config=execution_config,
    )
    pool_2 = RayRoundRobinActorPool(
        "torch-tensor-serde-overhead-2",
        1,
        resource_request,
        ExpressionsProjection([Identity2(daft.col("t")).alias("t")]),
        execution_config=execution_config,
    )

    ppm = PartialPartitionMetadata(num_rows=None, size_bytes=None)

    pool_1.setup()
    pool_2.setup()
    try:
        assert pool_1._actors is not None
        assert pool_2._actors is not None
        assert pool_1._actors[0]._actor_id != pool_2._actors[0]._actor_id

        input_ref = ray.put(MicroPartition.from_pydict({"t": rows}))

        _, warm_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[input_ref])
        ray.get(warm_ref)

        num_iters = 10
        start = time.perf_counter()
        cur_ref = input_ref
        for _ in range(num_iters):
            _, mid_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[cur_ref])
            _, cur_ref = pool_2.submit(partial_metadatas=[ppm], inputs=[mid_ref])
        out = ray.get(cur_ref)
        elapsed = time.perf_counter() - start

        bytes_per_row = tensors[0].numel() * tensors[0].element_size()
        total_bytes = bytes_per_row * n_rows * num_iters * 2
        mb_per_s = (total_bytes / (1024 * 1024)) / elapsed
        print(f"total_bytes: {total_bytes}")
        print(f"elapsed: {elapsed}")
        print(f"torch tensor object-store propagation throughput: {mb_per_s:.2f} MiB/s (elapsed={elapsed:.3f}s)")

        out_rows = out.to_pydict()["t"]
        assert len(out_rows) == len(rows)
        for got, expected in zip(out_rows, tensors):
            assert hasattr(got, "t")
            assert got.t.shape == expected.shape
            assert got.t.dtype == expected.dtype
            torch.testing.assert_close(got.t.sum(), expected.sum())
    finally:
        pool_1.teardown()
        pool_2.teardown()

# DAFT_ENABLE_PERF_TESTS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor.py -k test_tensor_object_store_serde_copy_overhead_between_actor_udfs_daft_tensor -s
"""
# default
DAFT_ENABLE_PERF_TESTS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor.py -k test_tensor_object_store_serde_copy_overhead_between_actor_udfs_daft_tensor -s

# zero-copy
DAFT_ENABLE_PERF_TESTS=1 RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1 pytest -q /home/wangzheyan/las-Daft/tests/series/test_tensor.py -k test_tensor_object_store_serde_copy_overhead_between_actor_udfs_daft_tensor -s
"""
def test_tensor_object_store_serde_copy_overhead_between_actor_udfs_daft_tensor() -> None:
    import os
    import time

    from tests.conftest import get_tests_daft_runner_name

    if get_tests_daft_runner_name() != "ray":
        pytest.skip("requires Ray runner")

    if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
        pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

    import ray

    import daft
    from daft import udf
    from daft.daft import PyDaftExecutionConfig, ResourceRequest
    from daft.datatype import DataType
    from daft.expressions import ExpressionsProjection
    from daft.recordbatch import MicroPartition
    from daft.runners.partitioning import PartialPartitionMetadata
    from daft.runners.ray_runner import RayRoundRobinActorPool

    shape = (512, 512)
    n_rows = 8
    arrs = [np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + i for i in range(n_rows)]

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
        "daft-tensor-serde-overhead-1",
        1,
        resource_request,
        ExpressionsProjection([Identity1(daft.col("t")).alias("t")]),
        execution_config=execution_config,
    )
    pool_2 = RayRoundRobinActorPool(
        "daft-tensor-serde-overhead-2",
        1,
        resource_request,
        ExpressionsProjection([Identity2(daft.col("t")).alias("t")]),
        execution_config=execution_config,
    )

    ppm = PartialPartitionMetadata(num_rows=None, size_bytes=None)

    pool_1.setup()
    pool_2.setup()
    try:
        input_ref = ray.put(MicroPartition.from_pydict({"t": arrs}))

        _, warm_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[input_ref])
        ray.get(warm_ref)

        num_iters = 10
        start = time.perf_counter()
        cur_ref = input_ref
        for _ in range(num_iters):
            _, mid_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[cur_ref])
            _, cur_ref = pool_2.submit(partial_metadatas=[ppm], inputs=[mid_ref])
        out = ray.get(cur_ref)
        elapsed = time.perf_counter() - start

        bytes_per_row = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        total_bytes = bytes_per_row * n_rows * num_iters * 2
        mb_per_s = (total_bytes / (1024 * 1024)) / elapsed
        print(f"total_bytes: {total_bytes}")
        print(f"elapsed: {elapsed}")
        print(f"daft tensor object-store propagation throughput: {mb_per_s:.2f} MiB/s (elapsed={elapsed:.3f}s)")

        out_arrs = out.to_pydict()["t"]
        assert len(out_arrs) == len(arrs)
        for got, expected in zip(out_arrs, arrs):
            np.testing.assert_equal(got, expected)
    finally:
        pool_1.teardown()
        pool_2.teardown()


# def test_tensor_object_store_serde_copy_overhead_between_actor_udfs_zero_copy_compare() -> None:
#     import json
#     import os
#     import subprocess
#     import sys
#     import textwrap

#     from tests.conftest import get_tests_daft_runner_name

#     if get_tests_daft_runner_name() != "ray":
#         pytest.skip("requires Ray runner")

#     if os.getenv("DAFT_ENABLE_PERF_TESTS") != "1":
#         pytest.skip("set DAFT_ENABLE_PERF_TESTS=1 to enable perf-style tests")

#     try:
#         import torch  # noqa: F401
#     except Exception:
#         pytest.skip("requires torch")

#     script = textwrap.dedent(
#         """
#         import json
#         import os
#         import resource
#         import time

#         mode = os.environ["ZERO_COPY_MODE"]

#         if mode == "driver_env_before_import":
#             os.environ["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] = "1"

#         import ray

#         runtime_env = None
#         if mode in ("runtime_env_only", "driver_env_before_import"):
#             runtime_env = {"env_vars": {"RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": "1"}}

#         ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), runtime_env=runtime_env)

#         import torch

#         import daft
#         from daft import udf
#         from daft.daft import PyDaftExecutionConfig, ResourceRequest
#         from daft.datatype import DataType
#         from daft.expressions import ExpressionsProjection
#         from daft.recordbatch import MicroPartition
#         from daft.runners.partitioning import PartialPartitionMetadata
#         from daft.runners.ray_runner import RayRoundRobinActorPool

#         class Wrapper:
#             def __init__(self, t):
#                 self.t = t

#         @ray.remote
#         def get_env_flag():
#             return os.environ.get("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS")

#         driver_flag = os.environ.get("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS")
#         worker_flag = ray.get(get_env_flag.remote())

#         shape = (8, 1024, 1024)
#         n_rows = 8
#         tensors = [torch.ones(shape, dtype=torch.float32) * float(i) for i in range(n_rows)]
#         rows = [Wrapper(t) for t in tensors]

#         @udf(return_dtype=DataType.python(), use_process=False)
#         class Identity1:
#             def __init__(self):
#                 pass

#             def __call__(self, t):
#                 return t

#         @udf(return_dtype=DataType.python(), use_process=False)
#         class Identity2:
#             def __init__(self):
#                 pass

#             def __call__(self, t):
#                 return t

#         execution_config = PyDaftExecutionConfig.from_env()
#         resource_request = ResourceRequest(num_cpus=1)

#         pool_1 = RayRoundRobinActorPool(
#             f"torch-tensor-serde-overhead-1-{mode}",
#             1,
#             resource_request,
#             ExpressionsProjection([Identity1(daft.col("t")).alias("t")]),
#             execution_config=execution_config,
#         )
#         pool_2 = RayRoundRobinActorPool(
#             f"torch-tensor-serde-overhead-2-{mode}",
#             1,
#             resource_request,
#             ExpressionsProjection([Identity2(daft.col("t")).alias("t")]),
#             execution_config=execution_config,
#         )

#         ppm = PartialPartitionMetadata(num_rows=None, size_bytes=None)

#         rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

#         pool_1.setup()
#         pool_2.setup()
#         try:
#             input_ref = ray.put(MicroPartition.from_pydict({"t": rows}))

#             _, warm_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[input_ref])
#             ray.get(warm_ref)

#             num_iters = 10
#             start = time.perf_counter()
#             cur_ref = input_ref
#             for _ in range(num_iters):
#                 _, mid_ref = pool_1.submit(partial_metadatas=[ppm], inputs=[cur_ref])
#                 _, cur_ref = pool_2.submit(partial_metadatas=[ppm], inputs=[mid_ref])
#             out = ray.get(cur_ref)
#             elapsed = time.perf_counter() - start

#             out_rows = out.to_pydict()["t"]
#             assert len(out_rows) == len(rows)
#             for got, expected in zip(out_rows, tensors):
#                 assert hasattr(got, "t")
#                 assert got.t.shape == expected.shape
#                 assert got.t.dtype == expected.dtype
#                 torch.testing.assert_close(got.t.sum(), expected.sum())
#         finally:
#             pool_1.teardown()
#             pool_2.teardown()

#         rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

#         bytes_per_row = tensors[0].numel() * tensors[0].element_size()
#         total_bytes = bytes_per_row * n_rows * num_iters * 2
#         mb_per_s = (total_bytes / (1024 * 1024)) / elapsed

#         print(
#             json.dumps(
#                 {
#                     "mode": mode,
#                     "driver_flag": driver_flag,
#                     "worker_flag": worker_flag,
#                     "elapsed_s": elapsed,
#                     "mbps": mb_per_s,
#                     "ru_maxrss_before": rss_before,
#                     "ru_maxrss_after": rss_after,
#                 }
#             )
#         )
#         """
#     )

#     def run_mode(mode: str) -> dict:
#         env = os.environ.copy()
#         env["ZERO_COPY_MODE"] = mode
#         env.setdefault("RAY_ADDRESS", "auto")
#         proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True)
#         for line in reversed(proc.stdout.splitlines()):
#             line = line.strip()
#             if line.startswith("{") and line.endswith("}"):
#                 return json.loads(line)
#         raise AssertionError(f"No JSON result found in stdout: {proc.stdout}\nSTDERR: {proc.stderr}")

#     baseline = run_mode("baseline")
#     runtime_env_only = run_mode("runtime_env_only")
#     driver_env_before_import = run_mode("driver_env_before_import")

#     print(
#         "baseline "
#         f"elapsed_s={baseline['elapsed_s']:.3f} mbps={baseline['mbps']:.2f} "
#         f"driver_flag={baseline['driver_flag']} worker_flag={baseline['worker_flag']}"
#     )
#     print(
#         "runtime_env_only "
#         f"elapsed_s={runtime_env_only['elapsed_s']:.3f} mbps={runtime_env_only['mbps']:.2f} "
#         f"driver_flag={runtime_env_only['driver_flag']} worker_flag={runtime_env_only['worker_flag']}"
#     )
#     print(
#         "driver_env_before_import "
#         f"elapsed_s={driver_env_before_import['elapsed_s']:.3f} mbps={driver_env_before_import['mbps']:.2f} "
#         f"driver_flag={driver_env_before_import['driver_flag']} worker_flag={driver_env_before_import['worker_flag']}"
#     )

#     assert baseline["mbps"] > 0
#     assert runtime_env_only["mbps"] > 0
#     assert driver_env_before_import["mbps"] > 0

#     assert runtime_env_only["worker_flag"] == "1"
#     assert driver_env_before_import["worker_flag"] == "1"
#     assert driver_env_before_import["driver_flag"] == "1"