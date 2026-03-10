# Repository Guidelines

## Project Structure & Module Organization
Core entrypoints are `src/train.py` and `src/eval.py`, both driven by Hydra
configs in `configs/`.
Put Lightning datamodules in `src/data/*_datamodule.py`, Lightning modules in
`src/models/*_module.py`, and shared helpers in `src/utils/`.
MAE-specific upstream-style components live under
`src/utils/masked_autoencoder/`; keep changes there focused and isolated.
Tests live in `tests/`, notebooks in `notebooks/`, shell helpers in `scripts/`,
local datasets in `data/`, and run artifacts in `logs/`.

## Build, Test, and Development Commands
Assume deployment and environment management happen remotely. Let the user handle
environment setup, dependency installation, and deployment steps unless they
explicitly ask for help with them. Treat the commands below as reference
commands, not default actions to run.

- `pip install -r requirements.txt`: install dependencies only when the user
  explicitly asks for local environment help.
- `make help`: list available make targets.
- `make test`: run the default non-slow pytest suite.
- `make test-full`: run the full test suite, including slow tests.
- `python src/train.py trainer=cpu logger=csv`: safe local smoke run without
  GPU or W&B.
- `python src/train.py experiment=mae_pretrain data.data_dir=/path/to/imagenet`:
  run MAE pretraining.
- `python src/eval.py ckpt_path=/path/to/model.ckpt`: evaluate a checkpoint.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and concise docstrings consistent with the
existing Lightning/Hydra code.
- Prefer single-line function signatures and function calls when they fit within
  120 columns.
- Keep compact signatures and calls compact. Do not split arguments one-per-line
  unless line length or readability requires it.
- If multiline formatting is needed, group multiple arguments per line when
  practical instead of using one argument per line.
- Keep `def name(` or `callee(` on the first line and pack that line as far as
  the 120-column limit allows before wrapping.
- When wrapping signatures or calls, prefer aligned continuation lines over
  hanging closing-paren layouts. Keep the closing `)` and `-> ReturnType` on
  the last argument line when they fit.
- Avoid adding trailing commas solely to force a vertically expanded layout.
- Keep string literals on one physical line when they fit within 120 columns.
  Avoid splitting one message into adjacent string literals unless needed for
  line length or readability.
- Multiline formatting is acceptable when arguments are being actively toggled
  or edited, when structured literals such as dictionaries read better
  expanded, or when expanded formatting materially improves maintainability.
Prefer `snake_case` for Python modules, functions, config keys, and test files.
Mirror config names to code paths where practical, for example
`configs/model/mae_pretrain.yaml` with `src/models/mae_pretrain_module.py`.
`pre-commit` is listed in dependencies, but no repo-local
`.pre-commit-config.yaml` is currently checked in.


## Testing Guidelines
Pytest is configured in `pyproject.toml`.
Name tests `test_*.py` and mark long-running cases with `@pytest.mark.slow`.
Keep fast coverage around config instantiation, train/eval entrypoints, and
datamodule behavior.
Use the fixtures in `tests/conftest.py` so tests write outputs to temporary
directories, not `logs/`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative summaries such as `cleaned up code` and
`added vis logging callback`.
Follow that style, keep each commit scoped, and mention the subsystem changed.
PRs should include the exact training or test command used, note any config
overrides or dataset-path requirements, and attach key metrics or screenshots
when behavior changes affect training outputs or logged visualizations.

## Security & Configuration Tips
Do not commit datasets, checkpoints, `logs/`, or secrets.
Keep machine-specific paths in untracked Hydra local config or environment
variables.
MAE pretraining requires an ImageNet-style directory and should not hardcode
absolute paths into shared configs.
