from __future__ import annotations

from daft.udf import cls, func, method
from typing import TYPE_CHECKING, Any
from daft.datatype import DataType
from daft.series import Series
from daft.runners import get_or_create_runner
from daft.expressions import col
import logging
import warnings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from daft.expressions import Expression
    from collections.abc import Callable
    from daft import DataFrame
    from daft.daft import IOConfig
    from ray.util.placement_group import PlacementGroup
    from ray.actor import ActorHandle
    import pathlib
    import numpy as np
    import ray

NUM_BUCKETS = "num_buckets"
KEY_COLUMN = "key_column"
NUM_CPUS = "num_cpus"
ENABLE_SCAN_TASK_SPLIT_AND_MERGE = "enable_scan_task_split_and_merge"
MAX_SOURCES_PER_SCAN_TASK = "max_sources_per_scan_task"
SCAN_TASKS_MAX_SIZE_BYTES = "scan_tasks_max_size_bytes"
SCAN_TASKS_MIN_SIZE_BYTES = "scan_tasks_min_size_bytes"
PLACEMENT_GROUP_READY_TIMEOUT_SECONDS = 10

class CheckpointActor:
    def __init__(self, bucket_id: int):
        self.bucket_id = bucket_id
        self.key_set: set[Any] = set()

    def add_keys(self, input_keys: list[Any]) -> None:
        self.key_set.update(input_keys)

    def filter(self, input_keys: list[Any]) -> "np.ndarray":  # noqa: UP037
        import numpy as np

        return np.array([input_key not in self.key_set for input_key in input_keys], dtype=bool)


# TODO: support native mode in future if needed
@cls(max_concurrency=1)
class CheckpointFilter:
    def __init__(self, num_buckets: int, actors_by_bucket: dict[int, ray.ActorHandle] | None = None):
        self.num_buckets = num_buckets
        self.actors_by_bucket: dict[int, ray.ActorHandle] = {}
        if actors_by_bucket is None:
            self._enabled = False
        else:
            self._enabled = True
            for idx in range(num_buckets):
                try:
                    self.actors_by_bucket[idx] = actors_by_bucket[idx]
                except KeyError as e:
                    raise RuntimeError(
                        f"CheckpointActor_{idx} not found. "
                        f"Please create actors before initializing CheckpointManager."
                    ) from e

    @method.batch(return_dtype=DataType.bool())
    def __call__(self, input: Series) -> Series:
        import numpy as np
        import ray

        if not self._enabled:
            return Series.from_numpy(np.full(len(input), True, dtype=bool))

        num_rows = len(input)
        if num_rows == 0:
            return Series.from_numpy(np.empty(0, dtype=bool))
        hash_arr = input.hash().to_arrow()
        hash_np = hash_arr.to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
        bucket_ids = (hash_np % np.uint64(self.num_buckets)).astype(np.int64, copy=False)

        futures = []
        row_indices_list: list[np.ndarray] = []
        row_order = np.argsort(bucket_ids, kind="stable")
        bucket_sorted = bucket_ids[row_order]
        run_starts = np.flatnonzero(np.r_[True, bucket_sorted[1:] != bucket_sorted[:-1]])
        run_ends = np.r_[run_starts[1:], len(bucket_sorted)]
        buckets_present = bucket_sorted[run_starts]

        # row_indices 是 input 里的行号（0-based），表示哪些 key 属于这个 bucket
        for bucket, start, end in zip(buckets_present, run_starts, run_ends):
            actor = self.actors_by_bucket[int(bucket)]
            row_indices = row_order[int(start) : int(end)]
            keys_subset = input.take(Series.from_numpy(row_indices, name="idx")).to_pylist() #TODO:耗时多
            futures.append(actor.filter.remote(keys_subset))         # TODO：耗时多
            row_indices_list.append(row_indices)
        try:
            results = ray.get(futures, timeout=300)         # TODO：耗时多
        except Exception as e:
            raise RuntimeError(f"CheckpointActor filter failed: {e}") from e

        final_result = np.full(num_rows, True, dtype=bool)
        for row_indices, subset_mask in zip(row_indices_list, results):
            final_result[row_indices] = subset_mask
        return Series.from_numpy(final_result)


# TODO: support native mode in future if needed
def try_apply_filter_and_collect(
    write_df: DataFrame,
    root_dir: str | pathlib.Path,
    io_config: IOConfig | None,
    checkpoint_config: dict[str, Any] | None,
    read_fn: Callable[..., DataFrame],
) -> DataFrame:
    """Apply checkpoint filter and collect, return collected DataFrame.

    Args:
        write_df: DataFrame built for writing.
        root_dir: Destination root directory.
        io_config: IO configuration for reading existing keys.
        checkpoint_config: Dict with key_column/num_buckets and optional num_cpus, or None.
        read_fn: Callable to read existing data (e.g., read_parquet/read_csv/read_json).

    Returns:
        collected DataFrame: Collected DataFrame(with filter inserted if any).
    """
    try:
        actor_handles: list[ActorHandle] | None = None
        placement_group: PlacementGroup | None = None

        if checkpoint_config is not None:
            (
                key_column,
                num_buckets,
                num_cpus,
                enable_scan_task_split_and_merge,
                max_sources_per_scan_task,
                scan_tasks_max_size_bytes,
                scan_tasks_min_size_bytes,
            ) = _validate_checkpoint_config(checkpoint_config)
            actor_handles, placement_group, checkpoint_filter = _prepare_checkpoint_filter(
                root_dir=root_dir,
                io_config=io_config,
                key_column=key_column,
                num_buckets=num_buckets,
                num_cpus=num_cpus,
                enable_scan_task_split_and_merge=enable_scan_task_split_and_merge,
                max_sources_per_scan_task=max_sources_per_scan_task,
                scan_tasks_max_size_bytes=scan_tasks_max_size_bytes,
                scan_tasks_min_size_bytes=scan_tasks_min_size_bytes,
                read_fn=read_fn,
            )
            if checkpoint_filter is not None:
                logger.info("Checkpoint filter enabled")
                write_df = write_df._insert_filter_after_source(checkpoint_filter)
            else:
                logger.info("Checkpoint filter is a no-op (no existing checkpoint data)")

        write_df.collect()
    finally:
        try:
            _cleanup_checkpoint_resources(actor_handles, placement_group)
        except Exception as e:
            warnings.warn(f"Unable to cleanup checkpoint resources: {e}")

    return write_df


# -----------------------------------------------------------------------------
# Internal helper function: prepare checkpoint filtering for write methods
# -----------------------------------------------------------------------------
def _validate_checkpoint_config(
    config: dict[str, Any],
) -> tuple[str, int, float, bool | None, int | None, int | None, int | None]:
    """Validates checkpoint configuration contains required fields with correct types.

    Args:
        config: Checkpoint configuration dictionary

    Returns:
        Tuple of (key_column, num_buckets, nums_cpu) with validated values

    Raises:
        ValueError: If required keys are missing or values are invalid
    """
    if not isinstance(config, dict):
        raise ValueError("checkpoint_config must be a dict")
    if KEY_COLUMN not in config:
        raise ValueError(f"checkpoint_config_dict must contain '{KEY_COLUMN}' key")
    key_column = config[KEY_COLUMN]
    if not isinstance(key_column, str):
        raise ValueError(f"'{KEY_COLUMN}' must be a string, got {type(key_column).__name__}")

    # Optional nums_buckets (int > 0), default to 4 if not provided
    num_buckets_obj = config.get(NUM_BUCKETS, 4)
    try:
        nb_float = float(num_buckets_obj)
    except Exception:
        raise ValueError(f"'{NUM_BUCKETS}' must be numeric (int/float), got {num_buckets_obj}")
    if not nb_float.is_integer() or nb_float <= 0:
        raise ValueError(f"'{NUM_BUCKETS}' must be a positive integer, got {num_buckets_obj}")
    num_buckets = int(nb_float)

    # Optional nums_cpu (float > 0), default to 1 if not provided
    num_cpus_obj = config.get(NUM_CPUS, 1)
    try:
        num_cpus = float(num_cpus_obj)
    except Exception:
        raise ValueError(f"'{NUM_CPUS}' must be numeric (int/float), got {num_cpus_obj}")
    if num_cpus <= 0:
        raise ValueError(f"'{NUM_CPUS}' must be > 0, got {num_cpus}")

    enable_scan_task_split_and_merge = config.get(ENABLE_SCAN_TASK_SPLIT_AND_MERGE, None)
    if enable_scan_task_split_and_merge is not None and not isinstance(enable_scan_task_split_and_merge, bool):
        raise ValueError(
            f"'{ENABLE_SCAN_TASK_SPLIT_AND_MERGE}' must be a bool, got {type(enable_scan_task_split_and_merge).__name__}"
        )

    max_sources_per_scan_task_obj = config.get(MAX_SOURCES_PER_SCAN_TASK, None)
    if max_sources_per_scan_task_obj is None:
        max_sources_per_scan_task = None
    else:
        try:
            ms_float = float(max_sources_per_scan_task_obj)
        except Exception:
            raise ValueError(
                f"'{MAX_SOURCES_PER_SCAN_TASK}' must be numeric (int/float), got {max_sources_per_scan_task_obj}"
            )
        if not ms_float.is_integer() or ms_float <= 0:
            raise ValueError(
                f"'{MAX_SOURCES_PER_SCAN_TASK}' must be a positive integer, got {max_sources_per_scan_task_obj}"
            )
        max_sources_per_scan_task = int(ms_float)

    scan_tasks_max_size_bytes_obj = config.get(SCAN_TASKS_MAX_SIZE_BYTES, None)
    if scan_tasks_max_size_bytes_obj is None:
        scan_tasks_max_size_bytes = None
    else:
        try:
            max_bytes = float(scan_tasks_max_size_bytes_obj)
        except Exception:
            raise ValueError(
                f"'{SCAN_TASKS_MAX_SIZE_BYTES}' must be numeric (int/float), got {scan_tasks_max_size_bytes_obj}"
            )
        if not max_bytes.is_integer() or max_bytes <= 0:
            raise ValueError(
                f"'{SCAN_TASKS_MAX_SIZE_BYTES}' must be a positive integer, got {scan_tasks_max_size_bytes_obj}"
            )
        scan_tasks_max_size_bytes = int(max_bytes)

    scan_tasks_min_size_bytes_obj = config.get(SCAN_TASKS_MIN_SIZE_BYTES, None)
    if scan_tasks_min_size_bytes_obj is None:
        scan_tasks_min_size_bytes = None
    else:
        try:
            min_bytes = float(scan_tasks_min_size_bytes_obj)
        except Exception:
            raise ValueError(
                f"'{SCAN_TASKS_MIN_SIZE_BYTES}' must be numeric (int/float), got {scan_tasks_min_size_bytes_obj}"
            )
        if not min_bytes.is_integer() or min_bytes <= 0:
            raise ValueError(
                f"'{SCAN_TASKS_MIN_SIZE_BYTES}' must be a positive integer, got {scan_tasks_min_size_bytes_obj}"
            )
        scan_tasks_min_size_bytes = int(min_bytes)

    return (
        key_column,
        num_buckets,
        num_cpus,
        enable_scan_task_split_and_merge,
        max_sources_per_scan_task,
        scan_tasks_max_size_bytes,
        scan_tasks_min_size_bytes,
    )


def _split_partitions_evenly(total: int, buckets: int) -> tuple[int, int]:
    """Return (base_len, remainder) for splitting total items into buckets."""
    base = total // buckets
    rem = total - buckets * base
    return base, rem


def _prepare_checkpoint_filter(
    root_dir: str | pathlib.Path,
    io_config: IOConfig | None,
    key_column: str,
    num_buckets: int,
    num_cpus: float,
    enable_scan_task_split_and_merge: bool | None,
    max_sources_per_scan_task: int | None,
    scan_tasks_max_size_bytes: int | None,
    scan_tasks_min_size_bytes: int | None,
    read_fn: Callable[..., DataFrame],
) -> tuple[list[ActorHandle], PlacementGroup | None, Expression | None]:
    """Build and return checkpoint resources.

    Returns:
        tuple[list[ActorHandle], PlacementGroup | None, Expression | None]:
            - actor_handles: created Ray actors for checkpoint filtering
            - placement_group: PG used to reserve/spread actor resources
            - checkpoint_filter_callable: Daft Expression used to filter input

    Notes:
        - If no existing partitions are found at `root_dir`, returns ([], None, None).
        - Raises RuntimeError if runner is not Ray.
    """
    if get_or_create_runner().name != "ray":
        raise RuntimeError("Checkpointing is only supported on Ray runner")

    logger.info(
        "Preparing checkpoint filter: root_dir=%s key_column=%s num_buckets=%s num_cpus=%s",
        str(root_dir),
        key_column,
        num_buckets,
        num_cpus,
    )

    # Build df_keys lazily; execution config for scan split/merge is applied at collect-time.
    df_keys = None
    try:
        df_keys = read_fn(path=str(root_dir), io_config=io_config)
        if key_column:
            df_keys = df_keys.select(key_column)
    except FileNotFoundError as e:
        warnings.warn(
            f"{root_dir} not found, checkpointing will not be supported because it's unnecessary. message: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Unable to read checkpoint at {root_dir}: {e}") from e

    if df_keys is None:
        return [], None, None

    # Create placement group and actors
    import ray
    from ray.exceptions import GetTimeoutError
    from ray.util.placement_group import placement_group

    pg = placement_group([{"CPU": num_cpus} for _ in range(num_buckets)], strategy="SPREAD")
    logger.info("Created checkpoint placement group: num_buckets=%s num_cpus=%s", num_buckets, num_cpus)
    try:
        # Wait for placement group to be ready with a timeout (seconds)
        ray.get(pg.ready(), timeout=PLACEMENT_GROUP_READY_TIMEOUT_SECONDS)
        logger.info("Checkpoint placement group is ready")
    except GetTimeoutError as timeout_err:
        # Best effort cleanup to avoid leaking PG
        try:
            ray.util.remove_placement_group(pg)
        except Exception as e:
            warnings.warn(f"Unable to remove placement group {pg}: {e}")
        raise RuntimeError(
            "Checkpoint resource reservation timed out. Try reducing 'num_buckets' and/or 'num_cpus', "
            "or ensure your Ray cluster has sufficient resources. "
            f"Error message: {timeout_err}"
        ) from timeout_err

    actor_handles: list[ActorHandle] = []
    actors_by_bucket: dict[int, ActorHandle] = {}
    try:
        logger.info("Creating checkpoint actors: count=%s", num_buckets)
        for i in range(num_buckets):
            actor = (
                ray.remote(max_concurrency=10)(CheckpointActor)
                .options(
                    num_cpus=num_cpus,
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                    ),
                )
                .remote(i)
            )
            actor_handles.append(actor)
            actors_by_bucket[i] = actor

        @func.batch(return_dtype=DataType.null())
        async def ingest_keys(
            input: Series,
            *,
            actors_by_bucket: dict[int, ActorHandle] = actors_by_bucket,
            num_buckets: int = num_buckets,
        ) -> Series:
            import asyncio
            import numpy as np
            import pyarrow as pa

            num_rows = len(input)
            if num_rows == 0:
                return Series.from_arrow(pa.nulls(0))

            keys = input.to_pylist()
            keys_np = np.asarray(keys, dtype=object)

            hash_arr = input.hash().to_arrow()
            hash_np = hash_arr.to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
            bucket_ids = (hash_np % np.uint64(num_buckets)).astype(np.int64, copy=False)

            row_order = np.argsort(bucket_ids, kind="stable")
            bucket_sorted = bucket_ids[row_order]
            run_starts = np.flatnonzero(np.r_[True, bucket_sorted[1:] != bucket_sorted[:-1]])
            run_ends = np.r_[run_starts[1:], len(bucket_sorted)]
            buckets_present = bucket_sorted[run_starts]

            futures = []
            for bucket, start, end in zip(buckets_present, run_starts, run_ends):
                actor = actors_by_bucket[int(bucket)]
                subset = keys_np[row_order[int(start) : int(end)]].tolist()
                print(f"in ingest_keys: bucket={bucket} start={start} end={end} num_keys={len(subset)}")
                futures.append(actor.add_keys.remote(subset))

            if futures:
                await asyncio.wait_for(asyncio.gather(*futures), timeout=300)

            return Series.from_arrow(pa.nulls(num_rows))

        from daft.context import execution_config_ctx
        with execution_config_ctx(
            enable_scan_task_split_and_merge=enable_scan_task_split_and_merge,
            max_sources_per_scan_task=max_sources_per_scan_task,
            scan_tasks_max_size_bytes=scan_tasks_max_size_bytes,
            scan_tasks_min_size_bytes=scan_tasks_min_size_bytes,
        ):
            df_keys.select(ingest_keys(col(key_column))).collect()
    except Exception as e:
        logger.exception("Failed to create all checkpoint actors")
        _cleanup_checkpoint_resources(actor_handles, pg)
        raise RuntimeError(f"Failed to create all checkpoint actors: {e}") from e
    finally:
        del df_keys

    checkpoint_filter = CheckpointFilter(num_buckets=num_buckets, actors_by_bucket=actors_by_bucket)
    checkpoint_filter_callable = checkpoint_filter(col(key_column))  # type: ignore
    return actor_handles, pg, checkpoint_filter_callable  # type: ignore


def _cleanup_checkpoint_resources(actor_handles: list[ActorHandle] | None, pg: PlacementGroup | None) -> None:
    """Cleanup checkpoint resources: terminate actors and remove placement group.

    Args:
        actor_handles: List of Ray ActorHandles to terminate.
        placement_group: The Ray placement group to remove.
    """
    import ray

    if actor_handles:
        for actor in actor_handles:
            try:
                ray.kill(actor)
            except Exception as e:
                warnings.warn(f"Unable to cleanup checkpoint resources: ray.kill failed: {e}")

    if pg:
        try:
            ray.util.remove_placement_group(pg)
        except Exception as e:
            warnings.warn(f"Unable to cleanup checkpoint resources: remove_placement_group failed: {e}")
