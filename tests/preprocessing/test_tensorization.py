"""Tests the tensorization script.

Note that this test relies on the tokenized shards from the tokenization test.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from . import TENSORIZATION_SCRIPT, single_stage_transform_tester
from .test_tokenization import WANT_EVENT_SEQS as TOKENIZED_SHARDS

WANT_NRTS = {
    f"{k.replace('event_seqs/', '')}.nrt": JointNestedRaggedTensorDict(
        v.select("time_delta_days", "code", "numeric_value").to_dict(as_series=False)
    )
    for k, v in TOKENIZED_SHARDS.items()
}


def test_tensorization():
    single_stage_transform_tester(
        transform_script=TENSORIZATION_SCRIPT,
        stage_name="tensorization",
        transform_stage_kwargs=None,
        input_shards=TOKENIZED_SHARDS,
        want_data=WANT_NRTS,
    )
