# Contributors Guide

Thank you for your interest in contributing to this project. The following
sections outline the expectations for contributors.

## Good Practice

- Discuss large changes with the maintainers before significant work begins.
- Keep commits focused and provide clear commit messages.
- Include tests for new functionality and bug fixes.
- Ensure documentation is updated for user-facing changes.

## Documentation Standards

Documentation should be written using Google style docstrings. Where examples
clarify behaviour, use doctests. Helpers like `yaml_disk` and `print_directory`
are provided via pytest plugins that extend the doctest namespace so examples
can remain concise. A global `conftest.py` further populates the namespace using
`pytest`'s `doctest_namespace` fixture, meaning you rarely need explicit imports
in doctests.

```python
>>> snippet = "a: 1"
>>> with yaml_disk(snippet) as path:
...     print_directory(path)
├── a
```

## Running Checks

Install the development dependencies and run the local quality checks:

```bash
pip install .[dev]
pre-commit run --all-files
```

Run the tests (including doctests):

```bash
pip install .[tests]
pytest -v
```

Only the files changed in a pull request are checked during continuous
integration. Passing these checks locally will ensure CI succeeds.
