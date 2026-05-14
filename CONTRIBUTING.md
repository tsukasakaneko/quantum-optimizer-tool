# Contributing

## Dev install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook,dev]"
pre-commit install
```

## Lint / format

```bash
ruff check .
ruff format .
```

## Tests

```bash
pytest
```

## Build

```bash
python -m build
```

`dist/` に sdist と wheel が生成される。

## Pull Requests

- 1 PR = 1 トピック。
- コミットメッセージは Conventional Commits 風（`feat:` / `fix:` / `chore:` / `docs:` ...）を推奨。
- `pre-commit` と `pytest` がローカルで通る状態で出す。
- CI（`.github/workflows/ci.yml`）が緑になることを確認。

## Issue / Discussion

機能追加や設計議論は Issue で先に合意してから実装するのが望ましい。
