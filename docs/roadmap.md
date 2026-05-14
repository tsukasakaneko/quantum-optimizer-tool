# Roadmap

## Vision

AWS 純正路線で完結する、AI エージェント向け量子最適化ツールを OSS で作る。
「自然言語の最適化問題」を「量子ソルバー」につなぐミドルレイヤーとして、
Strands / AgentCore エコシステムの量子入口になる。

## Technical Stack

| Layer | Component |
| --- | --- |
| LLM | Amazon Bedrock (Claude) |
| Agent | Strands Agents |
| Runtime | Bedrock AgentCore（将来） |
| Quantum | AWS Braket（Local Simulator / D-Wave） |
| Language | Python 3.11+ |
| License | Apache 2.0 |

## Core Features

1. 自然言語 / 構造化入力で最適化問題を受け取る
2. AI が QUBO 構造を自動生成（業務テンプレベース）
3. ペナルティ係数を自動チューニング（反復ループ）
4. Braket 経由で量子 / 古典ソルバーに投入
5. 結果を解釈してエージェントに返す

## Differentiation

| 対象 | 差別化 |
| --- | --- |
| LLM-QUBO（論文） | プロダクト化 + AWS 統合 |
| CODA MCP | 量子回路でなく最適化に特化 |
| Quant | Strands / AgentCore 深く統合 |
| OpenQAOA-Braket | エージェント呼び出し前提の UX |
| QUBO.jl | Python + 自然言語入力 |
| 共通 | AWS 純正経路 + 日本語対応 |

## Phases

### Phase 1: Braket 入門（1〜2 週）

- ローカルシミュレータで公式チュートリアル写経
- Jupyter Notebook で量子回路の基礎習得
- **本リポジトリ骨組みの作成（v0.0.1）**

### Phase 2: アニーリング体験（1 週）

- 最初の PoC 題材は **TSP（巡回セールスマン）**
- ローカル → D-Wave 実機で実行

### Phase 3: Strands tool 化（2 週）

- `@tool` デコレータで関数化
- Bedrock 経由で呼び出し確認

### Phase 4: AI 自動 QUBO 化（2 週）

- 自然言語入力 → QUBO 変換ループ
- ペナルティ係数自動調整

### Phase 5: OSS 公開・発信（1 週）

- GitHub 公開
- Qiita / Zenn 記事化
- AWS コミュニティで共有

## Post v0.1

- MCP 化（Strands 外からも使えるように）
- AgentCore Gateway 統合
- 業務テンプレ追加（シフト / 配送 / 選択 ...）
- 量子クラウド対応拡張（IBM, Fixstars 等）

## Open Items

- 発信プラットフォーム（Qiita vs Zenn vs 両方）
- TestPyPI への試験 upload タイミング
- ブランド名（プロジェクト識別名は `quantum-optimizer-tool` で確定）
