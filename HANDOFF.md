# HANDOFF — 開発状態の記録

このファイルは、リポジトリを初めて見る開発者・AI エージェントが
**「現在どこまで実装されているか / なぜこの設計か / 何が検証済みか / 次に何をすべきか」** を
コードと Git 履歴に照らして判断できるようにするための技術ドキュメントです。
ここに書かれている数値・状態は、記載時点(下記コミット)で **実際に実行して得た結果のみ** です。

- 対象ブランチ: `genspark_ai_developer`(PR #1 → `main`)
- 記載時点の直前コミット: `38bc97d`(Director Agent + README)。本ファイルを含むコミットで「検証・修正サイクル」を追加。
- 実測(記載時点):
  ```
  python3 -m pytest -q            -> 99 passed
    うち、Phase 1-2 の既存 7 ファイル -> 60 passed(変更なしで通過 = 後方互換)
    うち、tests/test_verification.py  -> 39 passed(今回追加)
  python3 -m mypy agentcomm       -> Success: no issues found in 24 source files (strict)
  python3 -m ruff check agentcomm tests examples -> All checks passed!
  examples/*.py (4 本)            -> すべて正常終了(オフライン、mock LLM)
  ```

---

## 1. 現在のアーキテクチャ(全体像)

```
User
 │ task_request
 ▼
DirectorAgent (BaseAgent)                     agentcomm/orchestration/director.py
 │  ├─ Planner.plan()        → Plan(DAG of SubTask)     orchestration/planner.py
 │  ├─ for each ready SubTask:  _solve_subtask()
 │  │     1 Question / 2 Hypothesis  … SubTask.reasoning に記録      orchestration/reasoning.py
 │  │     3 Execution  … _run_subtask()(既存 retry / fail-over)
 │  │     4 Critique   … Verifier.verify()(verifier=None なら省略)  orchestration/verification.py
 │  │     5 Refinement … RefinementPolicy.decide() → retry_same / reassign / gather_info
 │  │                    / replan / accept / give_up
 │  ├─ blocked → SKIPPED、Planner.replan()(既存)、Planner.synthesize()
 │  └─ reply task_result(metadata.plan = Plan.to_dict() 全記録)
 ▼
CommunicationLayer(Registry / Router / Transport / History)   ← 変更なし
 ▼
Worker Agent(LLMAgent など)/ Reviewer Agent                      ← 変更なし
```

| レイヤー | 場所 | 今回の変更 |
|---|---|---|
| 通信層(models / registry / router / transport / history / auth / layer) | `agentcomm/*.py` | **なし** |
| Agent 層(BaseAgent / EchoAgent / LLMAgent / ManagerAgent) | `agentcomm/agent.py` | **なし** |
| LLM Adapter 層 | `agentcomm/llm/` | **なし** |
| Orchestration 層 | `agentcomm/orchestration/` | **ここだけ**(下記 §3) |

Worker / Reviewer は Director の存在を知らず、`task_request` / `review_request` を受けて返信するだけです。
新しい通信経路は作っていません(全メッセージが `layer.router.request()` を通る)。

---

## 2. 役割分担(責務分離)

| コンポーネント | 責務 | 判断を含むか |
|---|---|---|
| **DirectorAgent** | サイクルの制御フローのみ。依存解決、Worker 選択、送受信、記録、上限管理 | しない(決定的) |
| **Planner** (`plan/replan/synthesize`) | 目標の分析・分解、追加タスク提案、最終統合 | する(LLM でも rule でも可) |
| **Worker Agent** | 実行して結果を返す。**自己評価はしない** | 実行のみ |
| **Verifier** (`verify → Critique`) | Worker とは**別の主体**として結果を検証 | する |
| **RefinementPolicy** (`decide → Refinement`) | 失敗した Critique に対する次の一手を選ぶ | する(既定は決定的ルール) |

Verifier の実装(全て `agentcomm/orchestration/verification.py`):

| 実装 | 検証主体 | 独立性 | 用途 |
|---|---|---|---|
| `RuleBasedVerifier` | 決定的ルール(空結果 / エラー文 / must_include 欠落 / 任意チェック) | 独立 | 安価な一次フィルタ |
| `LLMVerifier` | 任意の `LLMAdapter` に直接レビューさせる | 別モデル推奨 | Agent を増やさず LLM 検証 |
| `AgentVerifier` | **別 Agent**(role=`reviewer` 等)へ `review_request` を通信層経由で送る。Worker 自身は候補から除外 | 独立 | Test Agent → Review Agent 構成の土台 |
| `CompositeVerifier` | 上記を直列(最初の fail で停止) | — | 安価→高価の順で組合せ |

---

## 3. 今回追加・変更したファイルと状態

| ファイル | 種別 | 状態 | 内容 |
|---|---|---|---|
| `agentcomm/orchestration/reasoning.py` | **新規** | 実装済・テスト済 | `Question` / `Hypothesis` / `ExecutionRecord` / `Issue` / `Critique` / `Refinement` / `ReasoningTrace`、`Severity` / `IssueCategory` / `RefinementAction`。全て `to_dict/from_dict` |
| `agentcomm/orchestration/verification.py` | **新規** | 実装済・テスト済 | `Verifier` / `RefinementPolicy` Protocol、上表 4 Verifier、`DefaultRefinementPolicy`、レビュープロンプト、`feedback_text` |
| `agentcomm/orchestration/models.py` | 変更 | 実装済・テスト済 | `SubTask` に `verified_by: str \| None` と `reasoning: ReasoningTrace` を追加、`refinements` プロパティ(派生値)。`to_dict/from_dict` 後方互換(旧 JSON を読める) |
| `agentcomm/orchestration/director.py` | 変更 | 実装済・テスト済 | `verifier` / `refinement_policy` / `max_refinements` 引数追加。`_solve_subtask()`(サイクル)を新設し `_execute()` から呼ぶ。`_run_subtask()` は `extra` / `exclude` 引数と `ExecutionRecord` 記録を追加。`_critique` / `_fail_after_critique` / `_gather_info` 追加 |
| `agentcomm/orchestration/planner.py` | 変更 | 実装済・テスト済 | `render_results()` が最後の Critique(pass/fail、issue 一覧)を出力に含める(再計画・統合時に「なぜ却下されたか」を Planner が見られるようにするため)。それ以外の Planner ロジックは無変更 |
| `agentcomm/orchestration/__init__.py` | 変更 | 済 | 新シンボルの再エクスポート |
| `tests/test_verification.py` | **新規** | 39 テスト、全通過 | §5 参照 |
| `README.md` | 変更 | 済 | 「検証・修正サイクル」節を追加(本ファイルの要約) |
| `HANDOFF.md` | **新規** | — | 本ファイル |

**変更していないもの**(意図的): `agentcomm/agent.py`(`ManagerAgent` 含む)、`agentcomm/layer.py`、`agentcomm/router.py`、`agentcomm/llm/*`、既存テスト 7 ファイル、`examples/*`。

---

## 4. Question → Hypothesis → Execution → Critique → Refinement の実装

### 4.1 設計上の最重要事項: Internal Monologue は保存しない

`ReasoningTrace` は **Worker(LLM)の内部思考・chain-of-thought ではありません**。
Director 側が生成する **監査用の実行履歴** であり、各段階は「誰が・何を・いつ・どの message_id で」を要約した構造化データです。

- Worker へのプロンプトは通常の指示(+ 前工程の結果 + 却下時は構造化フィードバック)のみ。思考過程の出力は要求しない。
- Verifier へのプロンプトは「結論と根拠を固定 JSON で返せ。私的な推論は含めるな」(`review_system_prompt()`)。
- Reviewer の JSON に未知フィールド(例: scratchpad)があっても `Critique.from_dict` は既知フィールドしか取り込まない(テスト `test_worker_prompt_receives_only_structured_feedback` で保証)。
- `ExecutionRecord.result_summary` は 200 文字要約(`summarize()`)。全文は通信層の履歴(`message_id` 参照)にある。

### 4.2 各段階の対応

| 段階 | データ | 生成箇所 | 内容 |
|---|---|---|---|
| 1 Question | `ReasoningTrace.question: Question` | `DirectorAgent._record_question_and_hypothesis` | 指示の要約、入力となる依存 subtask id、要求(`metadata.requirements` / `must_include`)、goal / plan_id |
| 2 Hypothesis / Plan | `ReasoningTrace.hypothesis: Hypothesis` | 同上 | 方針(role へ委譲、依存結果の利用)、手順、期待成果。※ タスク分解そのものは従来通り `Planner.plan()` の `Plan.analysis` / `SubTask` |
| 3 Execution | `ReasoningTrace.executions: list[ExecutionRecord]` | `_run_subtask` | attempt、agent_id、request/result message_id、inputs_used、outcome(success/error/timeout/rejected)、要約、エラー |
| 4 Critique | `ReasoningTrace.critiques: list[Critique]` | `_critique` → `Verifier.verify` | verifier id、pass/fail、対象 attempt、summary、`Issue[]`(category / severity / summary / evidence / recommendation)、confidence、independent、review の message_id |
| 5 Refinement | `ReasoningTrace.refinements: list[Refinement]` | `RefinementPolicy.decide` | action、reason、after_attempt、instruction_delta(Worker に渡すフィードバック)、target_role、exclude_agents |

`SubTask.verified_by` = 最終的に pass を出した verifier id。`SubTask.refinements` = `len(reasoning.refinements)`。

### 4.3 制御フロー(`DirectorAgent._solve_subtask`)

```
record Question / Hypothesis
loop:
    _run_subtask(extra=feedback, exclude=agents)     # 既存 retry / fail-over(max_retries)
    if subtask not DONE or verifier is None: return  # 通信層レベルの失敗 or 検証なし
    critique = verifier.verify(...)                  # 例外は「検証失敗」Critique(passed=False, confidence=0)に変換
    if critique.passed: verified_by = ...; return
    decision = refinement_policy.decide(...)         # max_refinements を渡す
    ACCEPT   -> verified_by = ...; return
    GIVE_UP  -> FAILED; return
    REPLAN   -> FAILED; return                        # §6 参照
    GATHER_INFO -> 別 role に QUESTION を送り、回答を次の指示に添付
    REASSIGN    -> exclude に現 worker を追加
    RETRY_SAME  -> feedback を次の指示に添付
    status=READY にして loop 先頭へ
```

### 4.4 上限(無限ループ防止)— 3 段

| 上限 | 意味 | 既定 |
|---|---|---|
| `max_retries` | 1 回の Execution 内での通信レベル失敗(例外 / timeout / rejected)の再試行 | 1 |
| `max_refinements` | Critique fail に起因する再実行回数(subtask 単位)。超過で `GIVE_UP` | 2 |
| `max_rounds` | 計画ラウンド数(初回 + `replan`) | 5 |

### 4.5 `DefaultRefinementPolicy` の判断(決定的)

1. issue が全て `accept_below`(既定 medium)未満 → `ACCEPT`
2. `refinements_so_far >= max_refinements` → `GIVE_UP`
3. `missing_precondition` / `omission` かつ `info_role` 指定 → `GATHER_INFO`
4. `refinements_so_far + 1 >= replan_after`(既定 2)→ `REPLAN`
5. `worker_failure` / `hallucination` かつ 2 回目以降 → `REASSIGN`(現 worker を除外)
6. それ以外 → `RETRY_SAME`(フィードバック付き)

`replan_after=0` で REPLAN を無効化できる。Policy は Protocol なので差し替え可。

### 4.6 `verifier=None`(既定)の従来動作

- `_solve_subtask` は Execution 後に即 return。Critique / Refinement は一切実行されない。
- `PlanEvent` の列は従来と同一(`planned, dispatched, completed, finished`)。テスト `test_no_verifier_keeps_legacy_behaviour` と既存 60 テストで確認。
- 差分は `SubTask.reasoning` に Question / Hypothesis / ExecutionRecord が **追加で記録される** こと、`to_dict()` に `verified_by` / `reasoning` / `refinements` キーが増えること。`from_dict` は旧形式(これらのキーなし)も読める。

---

## 5. テスト(`tests/test_verification.py`、39 件、全て MockLLMAdapter でオフライン)

| 区分 | テスト(抜粋) | 検証内容 |
|---|---|---|
| 正常系 | `test_happy_path_question_plan_execution_critique_pass` | Q→P→E→C(pass)→Final。5 段階の記録、Reviewer Agent が検証、event 列 |
| 後方互換 | `test_no_verifier_keeps_legacy_behaviour` | verifier=None で従来の event 列・critique なし |
| 修正系 | `test_refinement_cycle_fail_then_retry_same_then_pass` | C(fail)→RETRY_SAME(構造化フィードバック)→再実行→C(pass) |
| 修正系 | `test_refinement_reassigns_to_another_agent` | RETRY_SAME → REASSIGN で別 Agent、exclude 記録 |
| 修正系 | `test_refinement_gather_info_from_other_role` / `..._falls_back_to_feedback_only` | 別 role へ QUESTION、`task_id=…:info`、role 不在時の継続 |
| 修正系 | `test_minor_issues_are_accepted` | low severity → ACCEPT |
| REPLAN | `test_replan_action_reaches_planner_replan_with_critique_context` ほか 4 件 | §6 |
| 失敗系 | `test_critique_keeps_failing_until_budget_exhausted` / `test_max_refinements_zero_…` | GIVE_UP、回数追跡 |
| 失敗系 | `test_worker_failure_before_critique_uses_existing_retry` / `test_reexecution_failure_after_refinement` | Worker 失敗・再実行失敗 |
| 失敗系 | `test_verifier_infrastructure_failure_is_recorded_not_passed` / `test_reviewer_agent_failure_…` / `test_reviewer_agent_returns_invalid_json` | Verifier 障害・不正 JSON は **pass にならない** |
| 失敗系 | `test_agent_verifier_no_reviewer_online` / `test_agent_verifier_never_lets_worker_review_itself` / `test_agent_verifier_prefers_reviewer_other_than_worker` | 独立性(自己レビュー禁止) |
| 失敗系 | `test_planner_failure_…` / `test_invalid_plans_are_rejected_before_execution`(unknown dep / cycle / self) / `test_empty_plan_is_invalid` | Planner 失敗・不正 Plan |
| 追跡 | `test_tracking_ids_worker_verifier_relationship_and_counts` | conversation_id / task_id 維持、message_id が履歴に実在、Worker↔Verifier 関係、attempts / refinements 回数、request metadata の `refinement` カウンタ |
| 追跡 | `test_director_reply_carries_reasoning_in_metadata` | 返信 `metadata.plan` に全記録、JSON 化・復元可 |
| 保護 | `test_worker_prompt_receives_only_structured_feedback` | Reviewer の余分なフィールドが Worker にも trace にも漏れない |
| 直列化 | `test_reasoning_trace_json_roundtrip` / `test_subtask_and_plan_roundtrip_with_reasoning` / `test_from_dict_tolerates_invalid_structured_data` | JSON 往復、不正値の許容(unknown category → OTHER 等)、旧形式互換 |
| 単体 | `test_rule_based_verifier_…` / `test_llm_verifier_…` / `test_composite_verifier_…` / `test_default_policy_…` / `test_render_results_includes_structured_critique_for_replanner` | 各コンポーネント |

---

## 6. REPLAN 経路の事実(要確認事項として明記)

**実装の事実**:
- `RefinementAction.REPLAN` を受けた `_solve_subtask` は、その SubTask を **`FAILED` にして return するだけ** です(`_fail_after_critique`、error 文字列に `[replan]` を付与)。`_solve_subtask` 自身は `planner.replan()` を呼びません。
- `planner.replan()` を呼ぶのは従来からある `_execute()` のループ末尾(DAG が finished になった後、`plan.rounds < max_rounds` の場合)**のみ** です。今回この呼び出し箇所は変更していません。
- したがって REPLAN は「**サブタスクを失敗扱いにして DAG ラウンドを終え、既存の replan 機構に判断を委ねる**」という間接的な接続です。Planner が代替タスクを返せば実行され(検証サイクル付き)、返さなければ plan は `FAILED` / `PARTIAL` で終わります。
- 追加の接続点として `render_results()`(replan / synthesize の入力)に最後の Critique を含めるようにしたので、`LLMPlanner.replan` は「なぜ却下されたか」を見て代替案を出せます。ただし **既定の `SequentialPlanner` / `StaticPlanner` の `replan()` は常に `[]` を返す** ため、それらを使う場合 REPLAN は GIVE_UP と実質同じ結果になります。

**テストで確認済み**(`tests/test_verification.py`):
- 代替タスクを返す Planner で REPLAN → 既存 `replan()` が呼ばれ、代替タスクが実行・検証される。`plan.rounds == 2`、event 順 `refinement → failed → replanned → dispatched`
- REPLAN で失敗した SubTask の依存先は既存の `blocked()` 処理で `SKIPPED`、Worker は起動されない
- `max_rounds=1` では `replan()` は呼ばれない
- Planner が `[]` を返す場合は plan `FAILED`
- Planner が不正な追加タスク(unknown dependency)を返した場合は `replan_rejected` で棄却

**未実装 / 検討余地**:
- REPLAN 時に「その SubTask を差し替える」「同 SubTask を別の指示で再生成する」といった **サブタスク単位の再計画 API** は Planner Protocol に存在しない。現状は plan 全体に対する `replan()`(追加のみ)に依存。
- REPLAN 直後にラウンドを打ち切って即 `replan()` を呼ぶのではなく、同ラウンドの他の ready タスクは通常どおり完了してから呼ばれる(既存挙動の踏襲。並列タスクを無駄にしないための意図的選択だが、要件次第で見直し余地あり)。

---

## 7. 実装済 / 未実装 / 要確認

### 実装済(テストで裏付けあり)
- 5 段階の構造化記録(`ReasoningTrace`)と JSON 往復、旧形式互換
- Worker と分離された Verifier(Rule / LLM / Agent / Composite)、自己レビュー禁止
- Critique の構造(pass/fail、issue の category / severity / summary / evidence / recommendation、confidence、independent、message_id)
- 9 種の検証観点(`IssueCategory` / `VERIFICATION_CHECKLIST`)を Verifier プロンプトと分類に反映
- Refinement 6 アクション(retry_same / reassign / replan / gather_info / accept / give_up)と `DefaultRefinementPolicy`
- 3 段の上限(`max_retries` / `max_refinements` / `max_rounds`)
- 既存 retry / fail-over / replan / blocked-skip の再利用(変更なし)
- conversation_id / task_id(`…:review`, `…:info` の階層)/ message_id の追跡
- Verifier 障害・不正 JSON・Reviewer 不在を「pass」に倒さない安全側の挙動
- `verifier=None` での従来動作維持(既存 60 テスト無変更で通過)

### 未実装
- **example / README のデモ**: `examples/` に Verifier を使うサンプルは **未作成**(既存 4 example は無変更で動作)
- Worker 側から Director への途中報告(`STATUS`)の活用
- サブタスク単位の再計画 API(§6)
- Verifier 結果に基づく Worker の信頼度スコアリング / 選択(`worker_selector` は critique を見ない)
- 複数 Verifier の多数決 / 重み付け(`CompositeVerifier` は直列のみ)
- Test Agent(実際にコードを実行してテストする Agent)— `IssueCategory.TEST_FAILURE` の分類と `AgentVerifier` の枕はあるが、実行系 Agent は存在しない
- 実 LLM(OpenAI / Anthropic / Google / xAI)での `LLMVerifier` / `LLMPlanner` の出力品質確認(鍵がない環境のため未実施。HTTP 層はモックで検証済)
- ネットワーク Transport(WebSocket / broker / A2A)、SQLite 履歴、at-least-once 配送(Phase 1 からの継続課題)

### 要確認
- `RuleBasedVerifier._ERROR_PATTERNS` はヒューリスティック(`I cannot`, `TODO`, 行末 `...` 等)。正当な結果を誤検知する可能性があり、実運用前に閾値・パターンの見直しが必要
- `DefaultRefinementPolicy` の既定値(`replan_after=2`, `accept_below=medium`)は実データで調整していない
- `AgentVerifier` は `review_request` の本文に system プロンプト相当のテキストを含めて送る(Reviewer が `LLMAgent` の場合、その system_prompt とは別に user メッセージとして渡る)。専用 `ReviewerAgent` を作るならこの契約を整理すべき
- `_gather_info` は追加情報を `extra_instruction` に連結して渡すだけで、`Question.inputs` には反映していない

---

## 8. 既知の問題

- `render_results()` の出力が Critique 分だけ長くなるため、`LLMPlanner.synthesize` / `replan` のプロンプト長が増える(大規模 plan での制御は未対応)
- `SubTask.to_dict()` に `refinements`(派生値)を含めているため、`from_dict` で `pop` している。スキーマとしてはやや冗長
- `ExecutionRecord.started_at` は `utc_now()` 既定であり、実際の送信直前の時刻とほぼ同じだが厳密には生成時刻
- テストの一部は `ScriptedVerifier`(テスト内二重)を使っており、`Verifier` Protocol の型検査は mypy 対象外(tests は mypy の対象に含めていない)

---

## 9. 次に検討すべき事項(優先順)

1. `examples/director_with_review.py` — Coding Agent → Reviewer Agent → Director → 修正依頼 のデモ(既存 `director_workflow.py` と同じ構成に `AgentVerifier` を足すだけで可能)
2. 実 LLM を 1 つ以上使った `LLMVerifier` / `LLMPlanner` の手動確認と、プロンプト調整
3. `Planner` に任意メソッド `refine_subtask(plan, subtask, critique)`(サブタスク単位の再計画)を追加するか検討。追加するなら Protocol の後方互換(既定実装 = 現在の挙動)を維持
4. Test Agent(サンドボックスでコード実行)と、その結果を `Issue(category=test_failure)` として返す契約
5. Verifier 履歴に基づく `worker_selector`(信頼度)と、Critique 統計の集計 API
6. Phase 1 からの継続: ネットワーク Transport、永続履歴、配送保証

---

## 10. 変更時の注意(影響範囲)

| 変更対象 | 影響 |
|---|---|
| `SubTask` のフィールド | `to_dict/from_dict`、`Plan.from_dict`、`render_results`、テスト全体。旧 JSON 互換を壊さないこと |
| `_run_subtask` の戻り/状態 | `_solve_subtask` は `st.status is DONE` で成功判定している。`base_attempt` を使った相対的 retry 計算に依存 |
| `PlanEvent.event` の文字列 | テストが event 列を直接比較している(`planned, dispatched, completed, critique, refinement, verified, failed, replanned, skipped, verification_failed, gather_info_failed, gathered_info, replan_rejected, replan_failed, synthesis_failed, finished`) |
| `review_system_prompt()` / `review_prompt()` | `AgentVerifier` と `LLMVerifier` が共有。`_result_under_review()` を使うテストは `"RESULT TO REVIEW"` 見出しに依存 |
| `DefaultRefinementPolicy` の順序 | §4.5 の判定順にテストが依存 |
| `Planner` Protocol | 3 メソッドのまま。増やす場合は既定実装を用意しないと `SequentialPlanner` / `StaticPlanner` / ユーザー実装が壊れる |
