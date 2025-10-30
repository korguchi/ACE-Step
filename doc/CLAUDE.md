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
**ファイル命名規則**:
```
{key}_vocal.mp3   # ボーカルトラック（必須）
{key}_inst.mp3    # インストトラック（accompanimentモード用）
{key}_mix.mp3     # ミックストラック（mixモード用、オプション）
{key}_prompt.txt  # カンマ区切りタグ（必須）
{key}_lyrics.txt  # 歌詞（オプション、なければ[instrumental]固定）
```

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

### 8 トラブルシューティング優先対処
**頻出問題と対処コマンド**:

| 問題 | 対処法 |
|------|-------|
| OOM | `batch_size: 1`, `accumulate_grad_batches: 8`, `lora.r: 64` に削減 |
| ボーカル漏れ | `reference_strength: 0.3` に削減、またはControlNet方式に切替 |
| プロンプト無視 | `target_modules` に `genre_embedder` を追加 |
| リズムずれ | `target_modules` に `lyric_proj` を追加、`reference_strength: 0.7` に増加 |
| Loss発散 | `learning_rate: 5e-5` に削減、`gradient_clip_val: 0.3` に強化 |

### 9 Claudeへの追加指示（このプロジェクト限定）
- コード生成時は **必ず型ヒント完備**（Pydantic BaseModel活用）
- 設定ファイルはJSON形式、ハードコード厳禁
- データセット関連コードは `AccompanimentDataset` クラスに集約
- 新規スクリプトは `scripts/` 配下に配置、`if __name__ == "__main__"` 必須
- LoRA保存は `transformers.save_lora_adapter()` のみ使用（フルモデル保存禁止）
- TensorBoardロギングは `self.log()` メソッドを使用、`on_step=True` 推奨
- Optunaの探索空間は `OptunaConfig.search_space` で定義、コード内にハードコード不可
- 推論スクリプトは `argparse` 使用、設定ファイルパスを `--config` で受け取る

### 10 ドキュメント更新ルール
- 実装方針変更時: `ACCOMPANIMENT_IMPLEMENTATION.md` を更新
- 新機能追加時: 該当セクションに使用例・コード例を追加
- トラブルシューティング追加時: 症状・原因・対処法を明記
- 設定スキーマ変更時: `config_schemas.py` のdocstringと実装方針書の両方を更新

### 11 Gitバージョン管理とブランチ戦略

#### 11.1 ブランチモデル
- **main**: 安定版ブランチ（マージ前にレビュー必須）
- **feature/***: 各フェーズの実装用ブランチ
  - `feature/phase1-foundation`: Phase 1（基盤インフラ）
  - `feature/phase2-image2image`: Phase 2（Image2Image方式）
  - `feature/phase3-controlnet`: Phase 3（ControlNet方式）
  - `feature/phase4-optuna`: Phase 4（Optuna統合）
  - `feature/phase5-inference`: Phase 5（推論・評価）
- **experiment/***: 実験用ブランチ（一時的、マージ不要）
  - `experiment/lora-rank-128`: LoRAランク128での検証
  - `experiment/ref-strength-0.3`: リファレンス強度0.3での実験

#### 11.2 コミットメッセージ規約
**フォーマット**: `[Phase番号] カテゴリ: 変更内容`

**カテゴリ**:
- `feat`: 新機能追加
- `fix`: バグ修正
- `refactor`: リファクタリング
- `docs`: ドキュメント更新
- `test`: テスト追加・修正
- `config`: 設定ファイル変更
- `chore`: その他（依存更新、ビルド設定等）

**例**:
```
[Phase1] feat: Add Pydantic config schemas for accompaniment system
[Phase2] feat: Implement Image2Image trainer with reference strength control
[Phase3] fix: Fix control encoder time embedding dimension mismatch
[Phase4] config: Add Optuna search space for ControlNet mode
[Phase5] docs: Add inference script usage examples to README
```

#### 11.3 マージ前チェックリスト
各フェーズのfeatureブランチをmainにマージする前に以下を実施:

1. **品質ゲート通過**（セクション6参照）
   ```bash
   uv run ruff check --fix acestep/ scripts/
   uv run ruff format acestep/ scripts/
   uv run ty check acestep/ scripts/
   uv run pytest tests/ -v
   ```

2. **短縮学習テスト**（100ステップで動作確認）
   ```bash
   uv run python scripts/train_image2image.py \
       --config config/accompaniment/test_config.json \
       --checkpoint_dir ./checkpoints \
       --devices 1 \
       --max_steps 100
   ```

3. **ドキュメント更新**
   - 新機能は `ACCOMPANIMENT_IMPLEMENTATION.md` に追記
   - 設定変更は該当セクションを更新
   - トラブルシューティング追加時は症状・原因・対処法を明記

4. **Git操作**
   ```bash
   git add .
   git commit -m "[PhaseN] feat: 変更内容の要約"
   git push origin feature/phaseN-description
   ```

#### 11.4 .gitignore 追加項目
伴奏生成プロジェクト固有の除外パターン:

```gitignore
# Optuna
optuna_*.db
optuna_*.db-shm
optuna_*.db-wal

# TensorBoard
exps/logs/*/events.out.tfevents.*
exps/logs/*/hparams.yaml

# チェックポイント（大容量）
exps/logs/*/checkpoints/*.safetensors
exps/logs/*/checkpoints/*.pth
*.lora_weights

# 実験出力
outputs/
comparison/
*.generated.mp3

# データセット（Gitで管理しない場合）
data/accompaniment_dataset/*.mp3
data/accompaniment_dataset/*.wav

# 一時ファイル
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.ty_cache/
```

#### 11.5 プルリクエスト（PR）テンプレート
各フェーズ完了時のPRには以下を含める:

```markdown
## [PhaseN] 機能名

### 変更内容
- 実装した主要機能のリスト
- 追加したファイル一覧

### 動作確認
- [ ] 品質ゲート通過（ruff, ty, pytest）
- [ ] 短縮学習テスト（100ステップ）成功
- [ ] TensorBoardログ確認
- [ ] ドキュメント更新済み

### 設定ファイル
- 追加した設定ファイルのパスと目的

### 注意事項
- 既存機能への影響
- 新規依存パッケージ
- 既知の制約事項

### レビュー観点
- コードの型安全性
- 設定のバリデーション
- エラーハンドリング
```

#### 11.6 実験結果の記録方針
- **実験ブランチ（experiment/*）の成果物は以下に記録**:
  1. **設定**: `config/accompaniment/experiments/` に日付付きで保存
     - 例: `experiment_20250130_rank128.json`
  2. **結果サマリ**: `doc/EXPERIMENT_LOG.md` に追記
     ```markdown
     ## 2025-01-30: LoRA Rank 128実験
     - **設定**: config/accompaniment/experiments/experiment_20250130_rank128.json
     - **結果**: Train Loss=0.0123, 音質スコア=4.2/5
     - **観察**: リズム同期は良好だが高周波帯域でノイズ
     - **次のアクション**: SSL係数を0.5に削減して再実験
     ```
  3. **生成サンプル**: GitHubにはアップせず、外部ストレージ（Google Drive等）のリンクを記録

#### 11.7 Claudeへの指示
- コード生成後は必ず**適切なコミットメッセージ案**を提示する
- Phase跨ぎの変更は分割コミットを推奨する
- `.gitignore` に追加すべき一時ファイルがあれば明示的に指摘する
- PRテンプレートの「動作確認」チェックリストを埋めるためのコマンド例を提供する
- Gitに関するコマンドは全てユーザーが実行するので自動で行わない．
