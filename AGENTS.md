# AGENT Guidelines

This repository expects contributors to follow the practices defined here. Any file in or below this
repository inherits these instructions.

## Good Contributor Practice

- Keep pull requests focused and prefer smaller, logically grouped changes.
- Ensure `pre-commit` hooks run cleanly before pushing or opening a pull request.
- Write tests alongside new features or bug fixes. Tests should be easy to read and avoid large
    set up code when possible.
- Use Google style docstrings for all public functions and classes. Provide examples via doctests
    where they aid understanding.

## Running Pre‑commit

First install the development dependencies and set up the hooks:

```bash
pip install .[dev]
pre-commit install
```

Run the hooks on all files with:

```bash
pre-commit run --all-files
```

During development you can run them only on changed files:

```bash
pre-commit run --files path1 [path2 ...]
```

The hooks format code with `ruff`, check docstrings, lint notebooks and YAML files, and generally
ensure style consistency.

## Running Tests

Tests are executed with `pytest` and include doctests. The default configuration is stored in
`pyproject.toml`. To run the full suite:

```bash
pip install .[tests]
pytest -v
```

Doctests are run via `pytest` so that the global `conftest.py` can populate the
`doctest_namespace` fixture. This allows commonly used utilities to be
pre-imported for better readability. The files `AGENTS.md` and `CONTRIBUTORS.md`
are excluded from doctest discovery.

Installed packages such as `yaml_disk` and `print_directory` register pytest
plugins that extend this namespace automatically, so you do not need to import
them or modify `conftest.py`.

### Doctest Helpers

Two utilities, [`yaml_disk`](https://github.com/mmcdermott/yaml_to_disk) and
[`print_directory`](https://github.com/mmcdermott/pretty-print-directory), are automatically added
to the doctest namespace when the packages are installed. They allow tests and documentation to set
up and display directory trees concisely.

```python
>>> contents = """
... foo:
...   bar.txt: "hello"
... """
>>> with yaml_disk(contents) as path:
...     print_directory(path)
├── foo
│   └── bar.txt
```

Use them in documentation and tests without explicit imports when possible. When additional doctest
features are required you may install `pytest-doctestplus`, though this dependency should be kept
optional to avoid bloat.
