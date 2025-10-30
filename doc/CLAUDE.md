## プロジェクト固有: ACE-Step 伴奏生成システム

このセクションは **ACE-Step リポジトリでの伴奏生成プロジェクト** 固有の規約を定義する。

### 1 プロジェクト概要
- **目標**: ボーカルトラックを入力として伴奏を生成するLoRAファインチューニングシステム
- **ベースモデル**: ACE-Step Diffusion Transformer
- **実装方式**: Image2Image方式（リファレンス混合）とControlNet方式（明示的条件付け）
- **詳細仕様**: `ACCOMPANIMENT_IMPLEMENTATION.md` を参照

### 2 設定管理方針
- **全設定はPydantic v2で型検証**: `acestep/config_schemas.py` にスキーマ定義
- **JSON設定ファイル**: `config/accompaniment/` 配下に方式別・実験別に配置
- **設定の継承**: 基本設定（base.json）から実験固有設定を派生させる
- **バリデーション**: 学習開始前に `TrainingConfig.model_validate()` で検証

### 3 データセット規約
**ファイル命名規則**（厳守）:
```
{key}_vocal.mp3   # ボーカルトラック（必須）
{key}_inst.mp3    # インストトラック（accompanimentモード用）
{key}_mix.mp3     # ミックストラック（mixモード用、オプション）
{key}_prompt.txt  # カンマ区切りタグ（必須）
{key}_lyrics.txt  # 歌詞（オプション、なければ[instrumental]固定）
```

**前処理要件**:
- 音源分離: Demucs / UVR 等で高品質分離
- サンプリングレート: 48kHz
- ラウドネス正規化: -14 LUFS 推奨
- ステレオ: モノラルは自動変換されるが品質低下の可能性

### 4 LoRA層選択戦略
**推奨target_modules**（用途別）:

| 用途 | target_modules | rank |
|------|---------------|------|
| **構造重視** | `to_q`, `to_k`, `to_v`, `to_out.0`, `lyric_proj` | 128 |
| **スタイル重視** | `to_q`, `to_k`, `to_v`, `to_out.0`, `genre_embedder` | 256 |
| **バランス** | 上記全て + `linear_q`, `linear_k`, `linear_v` | 192 |
| **最小実験** | `genre_embedder` のみ | 64 |

**非推奨**: `speaker_embedder`（ボーカルの音色特徴が混入するリスク）

### 5 実装フェーズと優先度
1. **Phase 1** (最優先): 基盤インフラ（Pydanticスキーマ、データセット拡張、ユーティリティ）
2. **Phase 2**: Image2Image方式実装
3. **Phase 3**: ControlNet方式実装（コントロールエンコーダ追加）
4. **Phase 4**: Optuna統合（自動ハイパーパラメータ探索）
5. **Phase 5**: 推論・評価スクリプト、比較HTML生成

### 6 品質ゲート（コード変更時の必須手順）
伴奏生成システムのコードを変更した場合、以下を**必ず**実行:

```bash
# 1. Lint & Format（自動修正）
uv run ruff check --fix acestep/ scripts/ trainer_accompaniment.py
uv run ruff format acestep/ scripts/ trainer_accompaniment.py

# 2. 型チェック
uv run ty check acestep/ scripts/

# 3. テスト（該当ファイルがある場合）
uv run pytest tests/test_accompaniment.py -v

# 4. 短縮学習テスト（100ステップで動作確認）
uv run python scripts/train_image2image.py \
    --config config/accompaniment/test_config.json \
    --checkpoint_dir ./checkpoints \
    --devices 1 \
    --max_steps 100
```

### 7 実験管理とロギング
- **TensorBoard必須**: すべての学習で `logger_dir` を指定し、損失・サンプル音声を記録
- **Optunaストレージ**: SQLiteファイル（`optuna_*.db`）はGit管理対象外（.gitignore追加）
- **チェックポイント命名**: `epoch={epoch}-step={step}_lora` 形式を維持
- **ベストモデル記録**: Optuna探索完了時に `*_best.json` として設定を自動保存

### 9.8 トラブルシューティング優先対処
**頻出問題と対処コマンド**:

| 問題 | 対処法 |
|------|-------|
| OOM | `batch_size: 1`, `accumulate_grad_batches: 8`, `lora.r: 64` に削減 |
| ボーカル漏れ | `reference_strength: 0.3` に削減、またはControlNet方式に切替 |
| プロンプト無視 | `target_modules` に `genre_embedder` を追加 |
| リズムずれ | `target_modules` に `lyric_proj` を追加、`reference_strength: 0.7` に増加 |
| Loss発散 | `learning_rate: 5e-5` に削減、`gradient_clip_val: 0.3` に強化 |

### 9.9 Claudeへの追加指示（このプロジェクト限定）
- コード生成時は **必ず型ヒント完備**（Pydantic BaseModel活用）
- 設定ファイルはJSON形式、ハードコード厳禁
- データセット関連コードは `AccompanimentDataset` クラスに集約
- 新規スクリプトは `scripts/` 配下に配置、`if __name__ == "__main__"` 必須
- LoRA保存は `transformers.save_lora_adapter()` のみ使用（フルモデル保存禁止）
- TensorBoardロギングは `self.log()` メソッドを使用、`on_step=True` 推奨
- Optunaの探索空間は `OptunaConfig.search_space` で定義、コード内にハードコード不可
- 推論スクリプトは `argparse` 使用、設定ファイルパスを `--config` で受け取る

### 9.10 ドキュメント更新ルール
- 実装方針変更時: `ACCOMPANIMENT_IMPLEMENTATION.md` を更新
- 新機能追加時: 該当セクションに使用例・コード例を追加
- トラブルシューティング追加時: 症状・原因・対処法を明記
- 設定スキーマ変更時: `config_schemas.py` のdocstringと実装方針書の両方を更新