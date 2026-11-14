import json
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
import rootutils
import torch
from mixins import MemTrackableMixin, TimeableMixin, add_mixin

from meds_torchdata.pytorch_dataset import MEDSPytorchDataset, MEDSTorchDataConfig

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

OUTPUT_DIR = root / "benchmark" / "outputs"

CNT_PREFIX = "Count: "


def tensor_size(a: torch.Tensor) -> int:
    return sys.getsizeof(a) + torch.numel(a) * a.element_size()


def to_val(k: str, v: Any) -> dict:
    match v:
        case list() as samples:
            vs = [to_val(k, val) for val in samples]
            cnt = 0
            sum_val = 0
            sum_val_sq = 0

            for v in vs:
                if "value" not in v:
                    raise ValueError(f"Value not found in {v}")

                if "extra" in v and v["extra"].startswith(CNT_PREFIX):
                    new_cnt = int(v["extra"][len(CNT_PREFIX) :])
                    cnt += new_cnt
                    sum_val += new_cnt * v["value"]

                    if "range" not in v:
                        raise ValueError(f"Range not found in {v} but count is")
                    v_stdev = v["range"]
                    v_var = v_stdev**2
                    v_sum_sq = (v_var + (v["value"] ** 2)) * new_cnt
                    sum_val_sq += v_sum_sq
                else:
                    cnt += 1
                    sum_val += v["value"]
                    sum_val_sq += v["value"] ** 2

            mean_val = sum_val / cnt
            std_val = np.sqrt((sum_val_sq - (sum_val**2) / cnt) / (cnt - 1))
            return {"value": mean_val, "range": std_val, "extra": f"{CNT_PREFIX}{cnt}"}
        case tuple() as stats if len(stats) == 3:
            mean_val, cnt, std_val = stats
            return {"value": mean_val, "range": std_val, "extra": f"{CNT_PREFIX}{cnt}"}
        case dict() as mem_stats if "metadata" in mem_stats and "peak_memory" in mem_stats["metadata"]:
            return {"value": mem_stats["metadata"]["peak_memory"]}
        case int() | float() as val:
            return {"value": val}
        case timedelta() as duration:
            return {"value": duration.total_seconds()}
        case _:
            raise ValueError(f"Unsupported type for {k}: {v}")


def summarize_output(out: dict) -> list[dict]:
    """Summarizes the output into a minimal format suitable for monitoring with github-actions-benchmark."""

    summary = []

    for single_key, name, unit in [
        ("epoch_durations", "Usage/Duration/Epoch", "seconds"),
    ]:
        summary.append({"name": name, "unit": unit, **to_val(single_key, out[single_key])})

    for nested_key, name, unit in [
        ("memory_stats", "Usage/Memory", "bytes"),
        ("duration_stats", "Usage/Duration", "seconds"),
        ("batch_sizes", "Outputs/BatchSize", "bytes"),
    ]:
        for k, v in out[nested_key].items():
            summary.append({"name": f"{name}/{k}", "unit": unit, **to_val(k, v)})

    return summary


def benchmark(dataset, batch_size: int, num_epochs: int = 1) -> tuple[dict[str, list[int]], list[timedelta]]:
    torch.manual_seed(1)

    dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)

    sizes = defaultdict(list)
    epoch_durations = []

    with dataset._track_memory_as("benchmark"):
        for _epoch in range(num_epochs):
            epoch_start = datetime.now(tz=UTC)
            for batch in dataloader:
                for k, v in batch.items():
                    sizes[k].append(tensor_size(v))
            epoch_durations.append(datetime.now(tz=UTC) - epoch_start)

    return sizes, epoch_durations


# Age-feature configurations to profile. `None` is the default (age features off), profiled on the
# standard benchmark dataset and written to the published `output_{...}.json`; the others run on the
# same data plus one birth event per subject and write `output_{...}_{name}.json` alongside it.
AGE_CONFIGS = {
    None: {},
    "age_off_with_births": {},  # Like-for-like baseline for the age configs below (same data).
    "include_age": {"include_age": True},
    "numeric_age_token": {"age_token_mode": "numeric"},
    "all_age_features": {"include_age": True, "age_token_mode": "quantized", "n_age_bins": 10},
}


@pytest.mark.parametrize("age_config", list(AGE_CONFIGS), ids=lambda c: c or "default")
@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("num_epochs", [5])
def test_profile(request, batch_size: int, max_seq_len: int, num_epochs: int, age_config: str | None):
    dataset_fixture = "benchmark_dataset" if age_config is None else "benchmark_dataset_with_births"
    benchmark_dataset = request.getfixturevalue(dataset_fixture)
    methods_to_track = ["__getitem__", "collate"]

    TrackableDataset: type[MEDSPytorchDataset] = add_mixin(  # noqa: N806
        add_mixin(
            MEDSPytorchDataset,
            TimeableMixin,
            dict.fromkeys(methods_to_track, TimeableMixin.TimeAs),
        ),
        MemTrackableMixin,
        {},
    )

    config = MEDSTorchDataConfig(
        tensorized_cohort_dir=benchmark_dataset,
        max_seq_len=max_seq_len,
        **AGE_CONFIGS[age_config],
    )

    D = TrackableDataset(config, split="train")
    batch_sizes, epoch_durations = benchmark(D, batch_size=batch_size, num_epochs=num_epochs)

    out = {}

    out["batch_sizes"] = batch_sizes
    out["epoch_durations"] = epoch_durations
    out["memory_stats"] = D._memory_stats
    out["duration_stats"] = D._duration_stats

    final_json_output = summarize_output(out)

    suffix = "" if age_config is None else f"_{age_config}"
    output_fp = OUTPUT_DIR / f"output_{batch_size}_{max_seq_len}_{num_epochs}{suffix}.json"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    output_fp.write_text(json.dumps(final_json_output))
