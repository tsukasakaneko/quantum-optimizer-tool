# Notebooks

Phase 1 の Braket 写経用ノートブック置き場。

## 起動

```bash
pip install -e ".[notebook]"
jupyter lab
```

`01_braket_hello.ipynb` から始める。

## AWS 認証についての注意

ローカルシミュレータ（`LocalSimulator`）のみを使う場合は AWS 認証情報は不要。

`AwsDevice` / D-Wave などのクラウド実機を使う場合は `aws configure` 済みで
あること。Braket 対応リージョン（`us-east-1`, `us-west-2` 等）を選ぶこと。

認証ファイルや API キーは絶対に notebook 内にハードコーディングしないこと。
`.gitignore` で `.aws/`, `credentials`, `*.pem` を弾いているが、流出防止の
ためコード側でも `os.environ` 経由でのみ参照すること。
