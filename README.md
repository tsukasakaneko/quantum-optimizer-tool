# quantum-optimizer-tool

自然言語で受けた最適化問題を、AWS Braket（量子）に橋渡しする
AI エージェント向けミドルレイヤー。

LLM が QUBO を組み立て、ペナルティ係数を反復チューニングし、Braket のローカル
シミュレータや D-Wave に投入して結果を解釈するまでをひとつのツールにまとめる
ことを目指す。Strands / Bedrock AgentCore エコシステムの「量子入口」に
なることを志向した OSS。

## Status

`v0.0.1` — Phase 1: Braket 入門中。骨組みのみで、QUBO・ソルバー・Strands
ツールは未実装。

## Architecture

```
User
  ↓ 自然言語の最適化問題
Amazon Bedrock (Claude)
  ↓
Strands Agents
  ↓ @tool 呼び出し
quantum_optimizer
  ├─ qubo/       QUBO 定式化
  ├─ templates/  業務テンプレ（TSP / ナップサック / シフト ...）
  ├─ solvers/    Braket ラッパ
  └─ tools/      Strands @tool エクスポート
  ↓
AWS Braket
  ├─ Local Simulator（開発・無料）
  └─ D-Wave（本番アニーリング）
```

将来的に Bedrock AgentCore からの起動・MCP 経由公開にも対応予定。

## Install

開発用 editable install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook,dev]"
```

ビルド成果物（wheel）から install することも可能:

```bash
pip install build
python -m build
pip install dist/quantum_optimizer-0.0.1-py3-none-any.whl
```

GitHub からの直 install:

```bash
pip install git+https://github.com/tsukasakaneko/quantum-optimizer-tool.git
```

## Phase 1 Quickstart

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/
```

`notebooks/01_braket_hello.ipynb` から AWS Braket SDK の写経を始める。

## AWS 認証セットアップ

Braket を使うには AWS アカウント側で Amazon Braket を有効化したうえで、
ローカルに認証情報を設定する:

```bash
aws configure
```

リージョンは Braket 対応の `us-east-1` / `us-west-2` などを推奨
（東京 `ap-northeast-1` は Braket 非対応）。

ローカルシミュレータのみであれば AWS 認証なしでも動作する。

## Roadmap

`docs/roadmap.md` を参照。

## License

Apache License 2.0 — `LICENSE` および `NOTICE` を参照。
