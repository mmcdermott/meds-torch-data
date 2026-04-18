"""`yaml_to_disk` `FileType` plugin for `.nrt` (`JointNestedRaggedTensorDict`) files.

Registered via the `[project.entry-points."yaml_to_disk.file_types"]` table in
`pyproject.toml`. yaml_to_disk discovers the plugin at runtime and routes any path ending in
`.nrt` through this class when materializing a YAML spec to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from yaml_to_disk.file_types.base import FileType


class NRTFile(FileType):
    """Validate and write `.nrt` (`JointNestedRaggedTensorDict`) files.

    Contents must be a mapping from tensor name to nested-list values accepted by
    `JointNestedRaggedTensorDict.__init__`.

    Examples:
        >>> import tempfile
        >>> from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
        >>> contents = {
        ...     "time_delta_days": [[float("nan"), 10.0], [float("nan"), 2.0]],
        ...     "code": [[[10, 11], [12]], [[20, 21], [22]]],
        ... }
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "out.nrt"
        ...     NRTFile.write(fp, contents)
        ...     sorted(JointNestedRaggedTensorDict(tensors_fp=fp).to_dense())
        ['code', 'dim1/mask', 'dim2/mask', 'time_delta_days']

        Non-mapping contents raise `TypeError`:

        >>> NRTFile.validate(["not", "a", "dict"])
        Traceback (most recent call last):
        ...
        TypeError: NRT contents must be a mapping of tensor name to nested lists; got list.

        `matches` picks up the `.nrt` extension:

        >>> NRTFile.matches(Path("x.nrt"))
        True
        >>> NRTFile.matches(Path("x.parquet"))
        False
    """

    extension: ClassVar[str] = ".nrt"

    @classmethod
    def validate(cls, contents: Any) -> None:
        if not isinstance(contents, dict):
            raise TypeError(
                f"NRT contents must be a mapping of tensor name to nested lists; "
                f"got {type(contents).__name__}."
            )
        JointNestedRaggedTensorDict(contents)

    @classmethod
    def write(cls, file_path: Path, contents: Any) -> None:
        cls.validate(contents)
        JointNestedRaggedTensorDict(contents).save(Path(file_path))
