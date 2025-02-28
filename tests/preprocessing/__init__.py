import json
import re
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from meds import subject_id_field, subject_splits_filepath
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal
from yaml import load as load_yaml

from .. import DEFAULT_CSV_TS_FORMAT, MEDS_PL_SCHEMA

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

TENSORIZATION_SCRIPT = "MTD_tensorize"
TOKENIZATION_SCRIPT = "MTD_tokenize"
PREPROCESS_SCRIPT = "MTD_preprocess"

FILE_T = pl.DataFrame | dict[str, Any] | str


def parse_meds_csvs(
    csvs: str | dict[str, str], schema: dict[str, pl.DataType] = MEDS_PL_SCHEMA
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Converts a string or dict of named strings to a MEDS DataFrame by interpreting them as CSVs."""

    default_read_schema = {**schema}
    default_read_schema["time"] = pl.Utf8

    def reader(csv_str: str) -> pl.DataFrame:
        cols = csv_str.strip().split("\n")[0].split(",")
        read_schema = {k: v for k, v in default_read_schema.items() if k in cols}
        return pl.read_csv(StringIO(csv_str), schema=read_schema).with_columns(
            pl.col("time").str.strptime(MEDS_PL_SCHEMA["time"], DEFAULT_CSV_TS_FORMAT)
        )

    if isinstance(csvs, str):
        return reader(csvs)
    else:
        return {k: reader(v) for k, v in csvs.items()}


def parse_shards_yaml(yaml_str: str, **schema_updates) -> pl.DataFrame:
    schema = {**MEDS_PL_SCHEMA, **schema_updates}
    return parse_meds_csvs(load_yaml(yaml_str.strip(), Loader=Loader), schema=schema)


MEDS_SHARDS = parse_shards_yaml(
    """
train/0: |-2
  subject_id,time,code,numeric_value
  239684,,EYE_COLOR//BROWN,
  239684,,HEIGHT,175.271115221764
  239684,"12/28/1980, 00:00:00",DOB,
  239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
  239684,"05/11/2010, 17:41:51",HR,102.6
  239684,"05/11/2010, 17:41:51",TEMP,96.0
  239684,"05/11/2010, 17:48:48",HR,105.1
  239684,"05/11/2010, 17:48:48",TEMP,96.2
  239684,"05/11/2010, 18:25:35",HR,113.4
  239684,"05/11/2010, 18:25:35",TEMP,95.8
  239684,"05/11/2010, 18:57:18",HR,112.6
  239684,"05/11/2010, 18:57:18",TEMP,95.5
  239684,"05/11/2010, 19:27:19",DISCHARGE,
  1195293,,EYE_COLOR//BLUE,
  1195293,,HEIGHT,164.6868838269085
  1195293,"06/20/1978, 00:00:00",DOB,
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
  1195293,"06/20/2010, 19:23:52",HR,109.0
  1195293,"06/20/2010, 19:23:52",TEMP,100.0
  1195293,"06/20/2010, 19:25:32",HR,114.1
  1195293,"06/20/2010, 19:25:32",TEMP,100.0
  1195293,"06/20/2010, 19:45:19",HR,119.8
  1195293,"06/20/2010, 19:45:19",TEMP,99.9
  1195293,"06/20/2010, 20:12:31",HR,112.5
  1195293,"06/20/2010, 20:12:31",TEMP,99.8
  1195293,"06/20/2010, 20:24:44",HR,107.7
  1195293,"06/20/2010, 20:24:44",TEMP,100.0
  1195293,"06/20/2010, 20:41:33",HR,107.5
  1195293,"06/20/2010, 20:41:33",TEMP,100.4
  1195293,"06/20/2010, 20:50:04",DISCHARGE,

train/1: |-2
  subject_id,time,code,numeric_value
  68729,,EYE_COLOR//HAZEL,
  68729,,HEIGHT,160.3953106166676
  68729,"03/09/1978, 00:00:00",DOB,
  68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,
  68729,"05/26/2010, 02:30:56",HR,86.0
  68729,"05/26/2010, 02:30:56",TEMP,97.8
  68729,"05/26/2010, 04:51:52",DISCHARGE,
  814703,,EYE_COLOR//HAZEL,
  814703,,HEIGHT,156.48559093209357
  814703,"03/28/1976, 00:00:00",DOB,
  814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,
  814703,"02/05/2010, 05:55:39",HR,170.2
  814703,"02/05/2010, 05:55:39",TEMP,100.1
  814703,"02/05/2010, 07:02:30",DISCHARGE,

tuning/0: |-2
  subject_id,time,code,numeric_value
  754281,,EYE_COLOR//BROWN,
  754281,,HEIGHT,166.22261567137025
  754281,"12/19/1988, 00:00:00",DOB,
  754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
  754281,"01/03/2010, 06:27:59",HR,142.0
  754281,"01/03/2010, 06:27:59",TEMP,99.8
  754281,"01/03/2010, 08:22:13",DISCHARGE,

held_out/0: |-2
  subject_id,time,code,numeric_value
  1500733,,EYE_COLOR//BROWN,
  1500733,,HEIGHT,158.60131573580904
  1500733,"07/20/1986, 00:00:00",DOB,
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
  1500733,"06/03/2010, 14:54:38",HR,91.4
  1500733,"06/03/2010, 14:54:38",TEMP,100.0
  1500733,"06/03/2010, 15:39:49",HR,84.4
  1500733,"06/03/2010, 15:39:49",TEMP,100.3
  1500733,"06/03/2010, 16:20:49",HR,90.1
  1500733,"06/03/2010, 16:20:49",TEMP,100.1
  1500733,"06/03/2010, 16:44:26",DISCHARGE,
    """
)


def assert_df_equal(want: pl.DataFrame, got: pl.DataFrame, msg: str = None, **kwargs):
    try:
        update_exprs = {}
        for k, v in want.schema.items():
            assert k in got.schema, f"missing column {k}."
            if kwargs.get("check_dtypes", False):
                assert v == got.schema[k], f"column {k} has different types."
            if v == pl.List(pl.String) and got.schema[k] == pl.List(pl.String):
                update_exprs[k] = pl.col(k).list.join("||")
        if update_exprs:
            want_cols = want.columns
            got_cols = got.columns

            want = want.with_columns(**update_exprs).select(want_cols)
            got = got.with_columns(**update_exprs).select(got_cols)

        assert_frame_equal(want, got, **kwargs)
    except AssertionError as e:
        pl.Config.set_tbl_rows(-1)
        raise AssertionError(f"{msg}:\nWant:\n{want}\nGot:\n{got}\n{e}") from e


def check_json(want: dict | Callable, got: dict, msg: str):
    try:
        match want:
            case dict():
                assert got == want, f"Want:\n{want}\nGot:\n{got}"
            case _ if callable(want):
                want(got)
            case _:
                raise ValueError(f"Unknown want type: {type(want)}")
    except AssertionError as e:
        raise AssertionError(f"{msg}: {e}") from e


def check_NRT_output(
    output_fp: Path,
    want_nrt: JointNestedRaggedTensorDict,
    msg: str,
):
    got_nrt = JointNestedRaggedTensorDict(tensors_fp=output_fp)

    # assert got_nrt.schema == want_nrt.schema, (
    #    f"Expected the schema of the NRT at {output_fp} to be equal to the target.\n"
    #    f"Wanted:\n{want_nrt.schema}\n"
    #    f"Got:\n{got_nrt.schema}"
    # )

    want_tensors = want_nrt.tensors
    got_tensors = got_nrt.tensors

    assert got_tensors.keys() == want_tensors.keys(), (
        f"{msg}:\n" f"Wanted:\n{list(want_tensors.keys())}\n" f"Got:\n{list(got_tensors.keys())}"
    )

    for k in want_tensors.keys():
        want_v = want_tensors[k]
        got_v = got_tensors[k]

        assert type(want_v) is type(
            got_v
        ), f"{msg}: Wanted {k} to be of type {type(want_v)}, got {type(got_v)}."

        if isinstance(want_v, list):
            assert len(want_v) == len(got_v), (
                f"Expected list {k} of the NRT at {output_fp} to be of the same length as the target.\n"
                f"Wanted:\n{len(want_v)}\n"
                f"Got:\n{len(got_v)}"
            )
            for i, (want_i, got_i) in enumerate(zip(want_v, got_v)):
                assert np.array_equal(want_i, got_i, equal_nan=True), (
                    f"Expected tensor {k}[{i}] of the NRT at {output_fp} to be equal to the target.\n"
                    f"Wanted:\n{want_i}\n"
                    f"Got:\n{got_i}"
                )
        else:
            assert np.array_equal(want_v, got_v, equal_nan=True), (
                f"Expected tensor {k} of the NRT at {output_fp} to be equal to the target.\n"
                f"Wanted:\n{want_v}\n"
                f"Got:\n{got_v}"
            )


def dict_to_hydra_kwargs(d: dict[str, str]) -> str:
    """Converts a dictionary to a hydra kwargs string for testing purposes.

    Args:
        d: The dictionary to convert.

    Returns:
        A string representation of the dictionary in hydra kwargs (dot-list) format.

    Raises:
        ValueError: If a key in the dictionary is not dot-list compatible.

    Examples:
        >>> print(" ".join(dict_to_hydra_kwargs({"a": 1, "b": "foo", "c": {"d": 2, "f": ["foo", "bar"]}})))
        a=1 b=foo c.d=2 'c.f=["foo", "bar"]'
        >>> from datetime import datetime
        >>> dict_to_hydra_kwargs({"a": 1, 2: "foo"})
        Traceback (most recent call last):
            ...
        ValueError: Expected all keys to be strings, got 2
        >>> dict_to_hydra_kwargs({"a": datetime(2021, 11, 1)})
        Traceback (most recent call last):
            ...
        ValueError: Unexpected type for value for key a: <class 'datetime.datetime'>: 2021-11-01 00:00:00
    """

    modifier_chars = ["~", "'", "++", "+"]

    out = []
    for k, v in d.items():
        if not isinstance(k, str):
            raise ValueError(f"Expected all keys to be strings, got {k}")
        match v:
            case bool() if v is True:
                out.append(f"{k}=true")
            case bool() if v is False:
                out.append(f"{k}=false")
            case None:
                out.append(f"~{k}")
            case str() | int() | float():
                out.append(f"{k}={v}")
            case dict():
                inner_kwargs = dict_to_hydra_kwargs(v)
                for inner_kv in inner_kwargs:
                    handled = False
                    for mod in modifier_chars:
                        if inner_kv.startswith(mod):
                            out.append(f"{mod}{k}.{inner_kv[len(mod):]}")
                            handled = True
                            break
                    if not handled:
                        out.append(f"{k}.{inner_kv}")
            case list() | tuple():
                v = list(v)
                v_str_inner = ", ".join(f'"{x}"' for x in v)
                out.append(f"'{k}=[{v_str_inner}]'")
            case _:
                raise ValueError(f"Unexpected type for value for key {k}: {type(v)}: {v}")

    return out


def run_command(
    script: Path | str,
    hydra_kwargs: dict[str, str],
    test_name: str,
    config_name: str | None = None,
    should_error: bool = False,
    do_use_config_yaml: bool = False,
    stage_name: str | None = None,
    do_pass_stage_name: bool = False,
):
    script = ["python", str(script.resolve())] if isinstance(script, Path) else [script]
    command_parts = script

    err_cmd_lines = []

    if config_name is not None and not config_name.startswith("_"):
        config_name = f"_{config_name}"

    if do_use_config_yaml:
        if config_name is None:
            raise ValueError("config_name must be provided if do_use_config_yaml is True.")

        conf = OmegaConf.create(
            {
                "defaults": [config_name],
                **hydra_kwargs,
            }
        )

        conf_dir = tempfile.TemporaryDirectory()
        conf_path = Path(conf_dir.name) / "config.yaml"
        OmegaConf.save(conf, conf_path)

        command_parts.extend(
            [
                f"--config-path={str(conf_path.parent.resolve())}",
                "--config-name=config",
                "'hydra.searchpath=[pkg://MEDS_transforms.configs]'",
            ]
        )
        err_cmd_lines.append(f"Using config yaml:\n{OmegaConf.to_yaml(conf)}")
    else:
        if config_name is not None:
            command_parts.append(f"--config-name={config_name}")
        command_parts.append(" ".join(dict_to_hydra_kwargs(hydra_kwargs)))

    if do_pass_stage_name:
        if stage_name is None:
            raise ValueError("stage_name must be provided if do_pass_stage_name is True.")
        command_parts.append(f"stage={stage_name}")

    full_cmd = " ".join(command_parts)
    err_cmd_lines.append(f"Running command: {full_cmd}")
    command_out = subprocess.run(full_cmd, shell=True, capture_output=True)

    command_errored = command_out.returncode != 0

    stderr = command_out.stderr.decode()
    err_cmd_lines.append(f"stderr:\n{stderr}")
    stdout = command_out.stdout.decode()
    err_cmd_lines.append(f"stdout:\n{stdout}")

    if should_error and not command_errored:
        if do_use_config_yaml:
            conf_dir.cleanup()
        raise AssertionError(
            f"{test_name} failed as command did not error when expected!\n" + "\n".join(err_cmd_lines)
        )
    elif not should_error and command_errored:
        if do_use_config_yaml:
            conf_dir.cleanup()
        raise AssertionError(
            f"{test_name} failed as command errored when not expected!\n" + "\n".join(err_cmd_lines)
        )
    if do_use_config_yaml:
        conf_dir.cleanup()
    return stderr, stdout


@contextmanager
def input_dataset(input_files: dict[str, FILE_T] | None = None):
    with tempfile.TemporaryDirectory() as d:
        input_dir = Path(d) / "input_cohort"
        cohort_dir = Path(d) / "output_cohort"

        for filename, data in input_files.items():
            fp = input_dir / filename
            fp.parent.mkdir(parents=True, exist_ok=True)

            match data:
                case pl.DataFrame() if fp.suffix == "":
                    data.write_parquet(fp.with_suffix(".parquet"), use_pyarrow=True)
                case pl.DataFrame() if fp.suffix in {".parquet", ".par"}:
                    data.write_parquet(fp, use_pyarrow=True)
                case pl.DataFrame() if fp.suffix == ".csv":
                    data.write_csv(fp)
                case dict() if fp.suffix == "":
                    fp.with_suffix(".json").write_text(json.dumps(data))
                case dict() if fp.suffix.endswith(".json"):
                    fp.write_text(json.dumps(data))
                case str():
                    fp.write_text(data.strip())
                case _ if callable(data):
                    data_str = data(
                        input_dir=str(input_dir.resolve()),
                        cohort_dir=str(cohort_dir.resolve()),
                    )
                    fp.write_text(data_str)
                case _:
                    raise ValueError(f"Unknown data type {type(data)} for file {fp.relative_to(input_dir)}")

        yield input_dir, cohort_dir


def check_outputs(
    cohort_dir: Path,
    want_outputs: dict[str, pl.DataFrame],
    assert_no_other_outputs: bool = True,
    **df_check_kwargs,
):
    all_file_suffixes = set()

    for output_name, want in want_outputs.items():
        if Path(output_name).suffix == "":
            output_name = f"{output_name}.parquet"

        file_suffix = Path(output_name).suffix
        all_file_suffixes.add(file_suffix)

        output_fp = cohort_dir / output_name

        files_found = [str(fp.relative_to(cohort_dir)) for fp in cohort_dir.glob("**/*{file_suffix}")]
        all_files_found = [str(fp.relative_to(cohort_dir)) for fp in cohort_dir.rglob("*")]

        if not output_fp.is_file():
            raise AssertionError(
                f"Wanted {output_fp.relative_to(cohort_dir)} to exist. "
                f"{len(files_found)} {file_suffix} files found with suffix: {', '.join(files_found)}. "
                f"{len(all_files_found)} generic files found: {', '.join(all_files_found)}."
            )

        msg = f"Expected {output_fp.relative_to(cohort_dir)} to be equal to the target"

        match file_suffix:
            case ".parquet":
                got_df = pl.read_parquet(output_fp, glob=False)
                assert_df_equal(want, got_df, msg=msg, **df_check_kwargs)
            case ".nrt":
                check_NRT_output(output_fp, want, msg=msg)
            case ".json":
                got = json.loads(output_fp.read_text())
                check_json(want, got, msg=msg)
            case _:
                raise ValueError(f"Unknown file suffix: {file_suffix}")

    if assert_no_other_outputs:
        all_outputs = []
        for suffix in all_file_suffixes:
            all_outputs.extend(list(cohort_dir.glob(f"**/*{suffix}")))
        assert len(want_outputs) == len(all_outputs), (
            f"Want {len(want_outputs)} outputs, but found {len(all_outputs)}.\n"
            f"Found outputs: {[fp.relative_to(cohort_dir) for fp in all_outputs]}\n"
        )


def single_stage_tester(
    script: str | Path,
    stage_name: str | None,
    stage_kwargs: dict[str, str] | None,
    do_pass_stage_name: bool = False,
    do_use_config_yaml: bool = False,
    want_outputs: dict[str, pl.DataFrame] | None = None,
    assert_no_other_outputs: bool = True,
    should_error: bool = False,
    config_name: str = "preprocess",
    input_files: dict[str, FILE_T] | None = None,
    df_check_kwargs: dict | None = None,
    test_name: str | None = None,
    do_include_dirs: bool = True,
    hydra_verbose: bool = True,
    stdout_regex: str | None = None,
    **pipeline_kwargs,
):
    if test_name is None:
        test_name = f"Single stage transform: {stage_name}"

    if df_check_kwargs is None:
        df_check_kwargs = {}

    if stage_kwargs is None:
        stage_kwargs = {}

    with input_dataset(input_files) as (input_dir, cohort_dir):
        for k, v in pipeline_kwargs.items():
            if type(v) is str and "{input_dir}" in v:
                pipeline_kwargs[k] = v.format(input_dir=str(input_dir.resolve()))
        for k, v in stage_kwargs.items():
            if type(v) is str and "{input_dir}" in v:
                stage_kwargs[k] = v.format(input_dir=str(input_dir.resolve()))

        pipeline_config_kwargs = {
            "hydra.verbose": hydra_verbose,
            **pipeline_kwargs,
        }

        if do_include_dirs:
            pipeline_config_kwargs["input_dir"] = str(input_dir.resolve())
            pipeline_config_kwargs["cohort_dir"] = str(cohort_dir.resolve())

        if stage_name is not None:
            pipeline_config_kwargs["stages"] = [stage_name]
        if stage_kwargs:
            pipeline_config_kwargs["stage_configs"] = {stage_name: stage_kwargs}

        run_command_kwargs = {
            "script": script,
            "hydra_kwargs": pipeline_config_kwargs,
            "test_name": test_name,
            "should_error": should_error,
            "config_name": config_name,
            "do_use_config_yaml": do_use_config_yaml,
        }

        if do_pass_stage_name:
            run_command_kwargs["stage"] = stage_name
            run_command_kwargs["do_pass_stage_name"] = True

        # Run the transform
        stderr, stdout = run_command(**run_command_kwargs)
        if should_error:
            return

        if stdout_regex is not None:
            regex = re.compile(stdout_regex)
            assert regex.search(stdout) is not None, (
                f"Expected stdout to match regex:\n{stdout_regex}\n" f"Got:\n{stdout}"
            )

        try:
            check_outputs(
                cohort_dir,
                want_outputs=want_outputs,
                assert_no_other_outputs=assert_no_other_outputs,
                **df_check_kwargs,
            )
        except Exception as e:
            raise AssertionError(
                f"Single stage transform {stage_name} failed -- {e}:\n"
                f"Script stdout:\n{stdout}\n"
                f"Script stderr:\n{stderr}\n"
            ) from e


def multi_stage_tester(
    scripts: list[str | Path],
    stage_names: list[str],
    stage_configs: dict[str, str] | str | None,
    do_pass_stage_name: bool | dict[str, bool] = True,
    want_outputs: dict[str, pl.DataFrame] | None = None,
    assert_no_other_outputs: bool = False,
    config_name: str = "preprocess",
    input_files: dict[str, FILE_T] | None = None,
    **pipeline_kwargs,
):
    with input_dataset(input_files) as (input_dir, cohort_dir):
        match stage_configs:
            case None:
                stage_configs = {}
            case str():
                stage_configs = load_yaml(stage_configs, Loader=Loader)
            case dict():
                pass
            case _:
                raise ValueError(f"Unknown stage_configs type: {type(stage_configs)}")

        match do_pass_stage_name:
            case True:
                do_pass_stage_name = {stage_name: True for stage_name in stage_names}
            case False:
                do_pass_stage_name = {stage_name: False for stage_name in stage_names}
            case dict():
                pass
            case _:
                raise ValueError(f"Unknown do_pass_stage_name type: {type(do_pass_stage_name)}")

        pipeline_config_kwargs = {
            "input_dir": str(input_dir.resolve()),
            "cohort_dir": str(cohort_dir.resolve()),
            "stages": stage_names,
            "stage_configs": stage_configs,
            "hydra.verbose": True,
            **pipeline_kwargs,
        }

        script_outputs = {}
        n_stages = len(stage_names)
        for i, (stage, script) in enumerate(zip(stage_names, scripts)):
            script_outputs[stage] = run_command(
                script=script,
                hydra_kwargs=pipeline_config_kwargs,
                do_use_config_yaml=True,
                config_name=config_name,
                test_name=f"Multi stage transform {i}/{n_stages}: {stage}",
                stage_name=stage,
                do_pass_stage_name=do_pass_stage_name[stage],
            )

        try:
            check_outputs(
                cohort_dir,
                want_outputs=want_outputs,
                assert_no_other_outputs=assert_no_other_outputs,
                check_column_order=False,
            )
        except Exception as e:
            raise AssertionError(f"{n_stages}-stage pipeline ({stage_names}) failed--{e}") from e


SHARDS = {
    "train/0": [239684, 1195293],
    "train/1": [68729, 814703],
    "tuning/0": [754281],
    "held_out/0": [1500733],
}

SPLITS_DF = pl.DataFrame(
    {
        subject_id_field: [239684, 1195293, 68729, 814703, 754281, 1500733],
        "split": ["train", "train", "train", "train", "tuning", "held_out"],
    }
)

MEDS_CODE_METADATA_CSV = """
code,code/n_occurrences,code/n_subjects,values/n_occurrences,values/sum,values/sum_sqd,description,parent_codes
,44,4,28,3198.8389005974336,382968.28937288234,,
ADMISSION//CARDIAC,2,2,0,,,,
ADMISSION//ORTHOPEDIC,1,1,0,,,,
ADMISSION//PULMONARY,1,1,0,,,,
DISCHARGE,4,4,0,,,,
DOB,4,4,0,,,,
EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",
EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",
EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",
HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
"""

MEDS_CODE_METADATA_SCHEMA = {
    "code": pl.Utf8,
    "code/n_occurrences": pl.UInt8,
    "code/n_subjects": pl.UInt8,
    "values/n_occurrences": pl.UInt8,
    "values/n_subjects": pl.UInt8,
    "values/sum": pl.Float32,
    "values/sum_sqd": pl.Float32,
    "values/n_ints": pl.UInt8,
    "values/min": pl.Float32,
    "values/max": pl.Float32,
    "description": pl.Utf8,
    "parent_codes": pl.Utf8,
    "code/vocab_index": pl.UInt8,
}


def parse_code_metadata_csv(csv_str: str) -> pl.DataFrame:
    cols = csv_str.strip().split("\n")[0].split(",")
    schema = {col: dt for col, dt in MEDS_CODE_METADATA_SCHEMA.items() if col in cols}
    df = pl.read_csv(StringIO(csv_str), schema=schema)
    if "parent_codes" in cols:
        df = df.with_columns(pl.col("parent_codes").cast(pl.List(pl.Utf8)))
    return df


MEDS_CODE_METADATA = parse_code_metadata_csv(MEDS_CODE_METADATA_CSV)


def remap_inputs_for_transform(
    input_code_metadata: pl.DataFrame | str | None = None,
    input_shards: dict[str, pl.DataFrame] | None = None,
    input_shards_map: dict[str, list[int]] | None = None,
    input_splits_map: dict[str, list[int]] | None = None,
    splits_fp: Path | str | None = subject_splits_filepath,
) -> dict[str, FILE_T]:
    unified_inputs = {}

    if input_code_metadata is None:
        input_code_metadata = MEDS_CODE_METADATA
    elif isinstance(input_code_metadata, str):
        input_code_metadata = parse_code_metadata_csv(input_code_metadata)

    unified_inputs["metadata/codes.parquet"] = input_code_metadata

    if input_shards is None:
        input_shards = MEDS_SHARDS

    for shard_name, df in input_shards.items():
        unified_inputs[f"data/{shard_name}.parquet"] = df

    if input_shards_map is None:
        input_shards_map = SHARDS

    unified_inputs["metadata/.shards.json"] = input_shards_map

    if input_splits_map is None:
        input_splits_map = SPLITS_DF

    if isinstance(input_splits_map, pl.DataFrame):
        input_splits_df = input_splits_map
    else:
        input_splits_as_df = defaultdict(list)
        for split_name, subject_ids in input_splits_map.items():
            input_splits_as_df[subject_id_field].extend(subject_ids)
            input_splits_as_df["split"].extend([split_name] * len(subject_ids))

        input_splits_df = pl.DataFrame(input_splits_as_df)

    if splits_fp is not None:
        # This case is added for error testing; not for general use.
        unified_inputs[splits_fp] = input_splits_df

    return unified_inputs


def single_stage_transform_tester(
    transform_script: str | Path,
    stage_name: str,
    transform_stage_kwargs: dict[str, str] | None,
    do_pass_stage_name: bool = False,
    do_use_config_yaml: bool = False,
    want_data: dict[str, pl.DataFrame] | None = None,
    want_metadata: pl.DataFrame | None = None,
    assert_no_other_outputs: bool = True,
    should_error: bool = False,
    df_check_kwargs: dict | None = None,
    **input_data_kwargs,
):
    if df_check_kwargs is None:
        df_check_kwargs = {}

    base_kwargs = {
        "script": transform_script,
        "stage_name": stage_name,
        "stage_kwargs": transform_stage_kwargs,
        "do_pass_stage_name": do_pass_stage_name,
        "do_use_config_yaml": do_use_config_yaml,
        "assert_no_other_outputs": assert_no_other_outputs,
        "should_error": should_error,
        "config_name": "preprocess",
        "input_files": remap_inputs_for_transform(**input_data_kwargs),
        "df_check_kwargs": df_check_kwargs,
    }

    want_outputs = {}
    if want_data:
        for data_fn, want in want_data.items():
            want_outputs[f"data/{data_fn}"] = want
    if want_metadata is not None:
        want_outputs["metadata/codes.parquet"] = want_metadata

    base_kwargs["want_outputs"] = want_outputs

    single_stage_tester(**base_kwargs)


def multi_stage_transform_tester(
    transform_scripts: list[str | Path],
    stage_names: list[str],
    stage_configs: dict[str, str] | str | None,
    do_pass_stage_name: bool | dict[str, bool] = True,
    want_data: dict[str, pl.DataFrame] | None = None,
    want_metadata: pl.DataFrame | None = None,
    **input_data_kwargs,
):
    base_kwargs = {
        "scripts": transform_scripts,
        "stage_names": stage_names,
        "stage_configs": stage_configs,
        "do_pass_stage_name": do_pass_stage_name,
        "assert_no_other_outputs": False,  # TODO(mmd): eventually fix
        "config_name": "preprocess",
        "input_files": remap_inputs_for_transform(**input_data_kwargs),
        "want_outputs": {**want_data, **want_metadata},
    }

    multi_stage_tester(**base_kwargs)
