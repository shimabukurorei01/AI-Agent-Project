# AI-Agent-Project — Agent-to-Agent Communication Layer

ChatGPT / Claude / Gemini / Grok などの「シングルエージェント」を、それぞれ独立した
AI エージェントとして同じプロジェクト内で動かし、相互にメッセージを送って
**議論・質問・回答・レビュー・タスク委譲** を行うための通信基盤 (`agentcomm`) です。

単なるチャットアプリではなく、将来的に Manager / Director / Research / Coding /
Reviewer Agent などへ拡張できる **Agent-to-Agent Communication Layer** を目的としています。

- 依存ライブラリ **ゼロ**(Python 3.11+ 標準ライブラリのみ。開発用に pytest / mypy / ruff)
- LLM に依存しない(通信層と LLM Adapter 層を完全分離)
- Agent 数に依存しない(後から追加・削除してもシステムは壊れない)
- 通信方式を差し替え可能(`Transport` 抽象。将来 WebSocket / message broker / A2A へ)
- API キーはコードに書かず `.env` から読む(`.env` は git-ignore 済み)
- 型安全(`mypy --strict` クリーン)、テスト付き(pytest 37件)

---

## 目次

1. [アーキテクチャ](#アーキテクチャ)
2. [ディレクトリ構成と各ファイルの役割](#ディレクトリ構成と各ファイルの役割)
3. [セットアップ・起動方法](#セットアップ起動方法)
4. [データ構造](#データ構造)
5. [Agent の追加方法](#agent-の追加方法)
6. [Agent 間通信の方法](#agent-間通信の方法)
7. [LLM の接続方法](#llm-の接続方法)
8. [テスト方法](#テスト方法)
9. [設計上の検討・トレードオフ](#設計上の検討トレードオフ)
10. [今後の拡張案](#今後の拡張案)

---

## アーキテクチャ

2 つの独立したレイヤーで構成されます。

```
                 ┌──────────────────────────────────────────────────────┐
   User ───────▶ │  Agent A          Agent B          Agent C  ...      │  Agent 層
                 │  (BaseAgent / LLMAgent / ManagerAgent)               │
                 └────┬──────────────────┬──────────────────┬───────────┘
                      │ send / request / reply / broadcast   │
                      ▼                  ▼                  ▼
                 ┌──────────────────────────────────────────────────────┐
                 │              CommunicationLayer (facade)             │
                 │  ┌───────────────┐ ┌──────────────┐ ┌─────────────┐ │  通信層
                 │  │ AgentRegistry │ │ MessageRouter│ │ HistoryStore│ │  (LLM を一切知らない)
                 │  │ id/role/caps  │ │ auth/route/  │ │ InMemory /  │ │
                 │  │ online 状態   │ │ timeout/reply│ │ JSONL       │ │
                 │  └───────────────┘ └──────┬───────┘ └─────────────┘ │
                 │                    ┌──────▼───────┐                  │
                 │                    │  Transport   │ ← InMemory(asyncio.Queue) │
                 │                    │  (抽象)      │   将来: WebSocket / Redis / A2A │
                 │                    └──────────────┘                  │
                 └──────────────────────────────────────────────────────┘

   Agent ──▶ LLMAdapter (抽象) ──▶ OpenAI / Anthropic / Google / xAI / Mock ...   LLM 層
                                                                          (通信層から独立)
```

- **Agent は他の Agent を直接参照しません。** 常に `CommunicationLayer` 経由で
  agent **id** に対してメッセージを送ります。そのため Agent の追加・削除で他が壊れません。
- **通信層は LLM を import しません。** LLM を使うかどうかは Agent の実装の自由です
  (`EchoAgent` は LLM なし、`LLMAgent` は任意の `LLMAdapter` を使用)。

### 3 段階の実装レベル(すべて動作確認済み)

| レベル | 内容 | 例 |
|---|---|---|
| MVP | Agent A → Layer → Agent B → 返信 → Agent A | `examples/roundtrip.py` |
| 拡張 | A/B/C/D 同時参加、broadcast、並列 request、1 体削除しても継続 | `examples/four_agents.py` |
| 階層 | Manager ─ Research / Coding / Reviewer / Testing Agent、role による委譲と集約 | `examples/manager_team.py` |

---

## ディレクトリ構成と各ファイルの役割

すべて **新規作成** です(既存ファイルの変更は `.gitignore` に `data/` を 1 行追加したのみ)。

```
AI-Agent-Project/
├── agentcomm/                 # ライブラリ本体
│   ├── __init__.py            # 公開 API の再エクスポート
│   ├── models.py              # Message / AgentInfo / MessageType / MessageStatus / AgentStatus
│   ├── errors.py              # 例外階層 (AgentNotFound, Offline, Timeout, RemoteAgentError, LLMError ...)
│   ├── config.py              # .env 読み込み・Settings・ロギング設定(依存なし)
│   ├── registry.py            # AgentRegistry: 登録/削除/状態/role・capability 検索
│   ├── transport.py           # Transport 抽象 + InMemoryTransport (asyncio.Queue)
│   ├── history.py             # HistoryStore 抽象 + InMemoryHistory + JsonlHistory(永続化)
│   ├── auth.py                # Authenticator: RegistryAuthenticator / TokenAuthenticator
│   ├── router.py              # MessageRouter: 認証・宛先解決・配送・履歴記録・request/reply 相関・timeout
│   ├── layer.py               # CommunicationLayer: 上記を束ねる facade
│   ├── agent.py               # BaseAgent / EchoAgent / LLMAgent / ManagerAgent
│   └── llm/                   # LLM Adapter 層(通信層から独立)
│       ├── base.py            # LLMAdapter 抽象 / ChatMessage / LLMResponse / HTTP ヘルパ
│       ├── mock.py            # MockLLMAdapter(テスト・オフラインデモ用)
│       ├── openai_compat.py   # OpenAI + xAI(Grok)+ OpenAI 互換 API
│       ├── anthropic.py       # Anthropic Messages API (Claude)
│       ├── google.py          # Google Gemini
│       └── factory.py         # "provider:model" 文字列から Adapter 生成、独自 provider 登録
├── examples/
│   ├── roundtrip.py           # MVP: A → B → A
│   ├── four_agents.py         # 4 Agent 同時参加・broadcast・削除耐性
│   └── manager_team.py        # Manager + 4 Worker の階層構成(JSONL 永続化つき)
├── tests/                     # pytest(外部 API には接続しない)
│   ├── test_models.py         # データ構造・JSON 往復・reply 型推論
│   ├── test_registry_history.py
│   ├── test_router.py         # 1対1 / broadcast / request-reply / timeout / 認証 / エラー通知
│   ├── test_agents.py         # A→B→A 往復、LLMAgent、Manager 階層、削除耐性、エラー耐性
│   └── test_llm.py            # 各 Adapter のペイロード生成・レスポンス解析(HTTP はモック)
├── pyproject.toml             # パッケージ定義、pytest/mypy/ruff 設定
├── .env.example               # API キーのテンプレート(.env 自体はコミット禁止)
└── .gitignore                 # .env / data/ などを除外
```

---

## セットアップ・起動方法

```bash
git clone https://github.com/shimabukurorei01/AI-Agent-Project.git
cd AI-Agent-Project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# API キー(任意。無くても mock で全て動きます)
cp .env.example .env
#   → OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY / XAI_API_KEY を記入
```

### サンプルの実行

```bash
# 1) MVP: Agent A → Agent B → Agent A(オフライン、mock LLM)
python examples/roundtrip.py

#    Agent B を実 LLM にする例
AGENT_B_MODEL=anthropic:claude-3-5-haiku-latest python examples/roundtrip.py

# 2) 4 Agent 同時参加(モデルは Agent ごとに指定可)
AGENT_A_MODEL=openai:gpt-4o-mini AGENT_B_MODEL=anthropic:claude-3-5-haiku-latest \
AGENT_C_MODEL=google:gemini-2.0-flash AGENT_D_MODEL=xai:grok-3-mini \
python examples/four_agents.py

# 3) Manager + Research / Coding / Reviewer / Testing
RESEARCH_MODEL=anthropic:claude-3-5-sonnet-latest CODING_MODEL=openai:gpt-4o \
python examples/manager_team.py      # 履歴は data/history.jsonl に永続化
```

環境変数の指定がない Agent はすべて `mock`(ネットワーク不要)になります。

---

## データ構造

### Message(`agentcomm/models.py`)

```json
{
  "message_id": "msg_1f3a9c2b7d4e",
  "conversation_id": "conv_74fec003905d",
  "task_id": "task_demo_001:coder",
  "sender": "research_agent",
  "receiver": "coding_agent",
  "message_type": "task_request",
  "content": "...",
  "timestamp": "2026-09-03T01:37:41.975+00:00",
  "reply_required": true,
  "in_reply_to": null,
  "status": "delivered",
  "metadata": {}
}
```

- `sender` / `receiver` は **agent id**(表示名ではない)。`receiver="*"` で broadcast。
- `message_type`: `chat, question, answer, task_request, task_accepted, task_result,
  task_rejected, review_request, review_result, broadcast, status, error, ack`
- `status`: `created → sent → delivered → processed / replied / failed / timeout`
- `Message.reply(content)` で `conversation_id` / `task_id` / `in_reply_to` を引き継いだ返信を生成
  (`question→answer`, `task_request→task_result`, `review_request→review_result` を自動推論)。
- `to_dict()` / `from_dict()` で JSON 化可能(ネットワーク転送を想定)。

### AgentInfo

```json
{
  "id": "agent_001",
  "name": "Research Agent",
  "role": "researcher",
  "model": "anthropic:claude-3-5-sonnet-latest",
  "capabilities": ["research", "analysis"],
  "endpoint": null,
  "status": "online"
}
```

`status`: `online / busy / offline / error`。`endpoint` はネットワーク化時の URL / queue 名用に予約。

---

## Agent の追加方法

### 1. 既存クラスを使う(最短)

```python
from agentcomm import AgentInfo, CommunicationLayer, LLMAgent
from agentcomm.llm import create_adapter

layer = CommunicationLayer()
agent = LLMAgent(
    layer,
    AgentInfo(id="reviewer_agent", name="Reviewer", role="reviewer",
              capabilities=["review", "security"]),
    create_adapter("anthropic:claude-3-5-sonnet-latest"),   # or "mock"
    system_prompt="You review code for security issues.",  # 省略可
)
await agent.start()      # 登録 + 受信ループ開始 + online
...
await agent.stop()       # 受信ループ停止 + 登録解除(他 Agent は影響を受けない)
```

### 2. 独自の振る舞いを持つ Agent を作る

`BaseAgent` を継承し `handle()` を実装するだけです。

```python
from agentcomm import AgentInfo, BaseAgent, Message, MessageType

class TriageAgent(BaseAgent):
    async def handle(self, message: Message) -> None:
        if message.message_type == MessageType.ERROR:
            self.log.warning("error from %s: %s", message.sender, message.content)
            return
        if "bug" in message.content.lower():
            result = await self.delegate("coding_agent", message.content)  # タスク委譲
            await self.reply(message, f"Fixed by coder: {result.content}")
        elif message.reply_required:
            await self.reply(message, "Not a bug. Closing.")

agent = TriageAgent(layer, AgentInfo(id="triage", name="Triage", role="triage"))
await agent.start()
```

`handle()` 内の例外はループを止めず、送信元に `ERROR` メッセージとして返送されます。
`handle_timeout=` を指定するとハンドラの実行時間にも上限をかけられます。

### 3. Agent オブジェクトなしで参加する(外部プロセス想定)

```python
await layer.register(AgentInfo(id="external", name="External", role="tool"))
msg = await layer.receive("external", timeout=10)   # 自分で受信
await layer.reply(msg, "done")
```

---

## Agent 間通信の方法

```python
# 1対1(fire-and-forget)
await agent_a.send("agent_b", "FYI: spec updated", message_type=MessageType.CHAT)

# 質問して返信を待つ(timeout 付き。Future で相関、mailbox には流れない)
reply = await agent_a.ask("agent_b", "What is A2A?", timeout=30)

# タスク委譲(task_id を自動生成、message_type=task_request)
result = await agent_a.delegate("coding_agent", "Implement a rate limiter", task_id="task_42")

# レビュー依頼
review = await agent_a.ask("reviewer_agent", code, message_type=MessageType.REVIEW_REQUEST)

# ブロードキャスト(自分以外の online Agent 全員)
delivered_ids = await agent_a.broadcast("Kick-off!")

# 返信(conversation_id / task_id / in_reply_to を自動継承)
await agent_b.reply(incoming, "Here is my answer")

# 会話履歴
layer.conversation(conversation_id)   # 会話単位
layer.task_history(task_id)           # タスク単位
layer.agent_history("agent_b")        # Agent 単位
```

### エラー・タイムアウトの挙動

| 状況 | 挙動 |
|---|---|
| 未登録の receiver | `AgentNotFoundError`、履歴に `failed` で記録 |
| offline の receiver | `AgentOfflineError` |
| 未登録の sender(なりすまし) | `UnauthorizedSenderError` |
| `request()` に返信なし | `MessageTimeoutError`、履歴に `timeout` |
| 相手の `handle()` が例外 | 相手から `ERROR` メッセージ。`request()` 中なら即 `RemoteAgentError`(timeout を待たない) |
| Agent 削除後に送信 | `AgentNotFoundError`。他 Agent の通信は継続 |

すべてのメッセージと状態遷移は `logging`(`agentcomm.*` ロガー)と `HistoryStore` に記録されます。

### Manager 階層(委譲と集約)

```python
manager = ManagerAgent(layer, AgentInfo(id="manager", name="Manager", role="manager"),
                       adapter=create_adapter("openai:gpt-4o"),  # 省略可: 省略時は結果を連結
                       task_timeout=120)
# 「role → 指示」の plan を渡すと、その role の online Agent に並列で委譲し結果を集約
result = await user.ask("manager", "Build a rate limiter",
                        message_type=MessageType.TASK_REQUEST,
                        metadata={"plan": {
                            "researcher": "Research algorithms",
                            "coder":      "Implement in Python",
                            "reviewer":   "List review criteria",
                            "tester":     "Propose a test plan",
                        }})
print(result.content)             # 集約レポート
print(result.metadata["results"]) # role ごとの生結果(欠員 / timeout は "ERROR: ..." で報告)
```

---

## LLM の接続方法

`"provider:model"` 文字列で指定します。キーは環境変数(`.env`)から読まれます。

| provider 名 | 実装 | 環境変数 |
|---|---|---|
| `openai` / `chatgpt` | OpenAI Chat Completions | `OPENAI_API_KEY`, `OPENAI_BASE_URL`(任意) |
| `anthropic` / `claude` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `google` / `gemini` | Gemini generateContent | `GOOGLE_API_KEY` |
| `xai` / `grok` | xAI(OpenAI 互換) | `XAI_API_KEY` |
| `mock` | オフライン・テスト用 | なし |

```python
from agentcomm.llm import create_adapter, register_provider, LLMAdapter

adapter = create_adapter("google:gemini-2.0-flash")

# 独自 provider(ローカル LLM など)の追加
class OllamaAdapter(LLMAdapter): ...
register_provider("ollama", lambda model, **kw: OllamaAdapter(model or "llama3", **kw))
create_adapter("ollama:llama3")

# OpenAI 互換サーバはそのまま使える
create_adapter("openai:llama3", base_url="http://localhost:11434/v1", api_key="none")
```

`LLMAdapter` の契約は `complete(messages, system=..., temperature=..., max_tokens=...) -> LLMResponse`
の 1 メソッドのみです。

---

## テスト方法

```bash
pytest -q                    # 37 tests、外部 API 不要(0.6 秒程度)
mypy agentcomm               # --strict(pyproject.toml で設定)
ruff check agentcomm tests examples
```

主なテスト内容:

- `test_agents.py::test_roundtrip_a_to_b_to_a` — **Agent A → B → A の往復通信**(MVP 要件)
- 4 Agent 同時参加・broadcast・並列 request・1 体削除後の継続
- Manager が role で委譲・集約、欠員 / timeout の報告
- LLM 障害時に `ERROR` 返送 & ループ生存、ハンドラ timeout
- request/reply 相関、timeout、認証(Registry / Token)、履歴クエリ、JSONL 永続化と replay
- 各 LLM Adapter のリクエスト生成・レスポンス解析(HTTP はモック、キー欠落時のエラー)

---

## 設計上の検討・トレードオフ

実装前に検討した 10 項目と、採用した判断です。

| # | 項目 | 判断 | トレードオフ / 補足 |
|---|---|---|---|
| 1 | 通信方式 | **asyncio + in-process mailbox(Queue)** を `Transport` 抽象の背後に置く | 最小・高速・デバッグ容易。プロセス分散は不可 → 抽象を実装して差替え(§今後) |
| 2 | データ構造 | `dataclass(slots=True)` + `str` Enum、`to_dict/from_dict` | pydantic 等を避け依存ゼロ。バリデーションは型ヒント + mypy に委ねる |
| 3 | Agent Registry | 記述子(`AgentInfo`)のみを保持し、Agent オブジェクトは保持しない | 他プロセスの Agent も同じ registry に登録できる。ライフサイクルは Agent 側が管理 |
| 4 | Message Router | 認証→宛先解決→履歴記録→配送→reply 相関を 1 箇所に集約 | `request()` の返信は Future に直接渡し mailbox には入れない(二重処理防止)。broadcast は receiver ごとにコピーを生成し `message_id:receiver` で追跡 |
| 5 | LLM Adapter | 単一メソッド `complete()`、標準ライブラリ `urllib` を `asyncio.to_thread` で非同期化 | SDK 不要・依存ゼロ。streaming / tool-calling は未対応(必要なら Adapter を拡張) |
| 6 | Memory / 履歴 | 共有 `HistoryStore` を単一の真実とし、`LLMAgent` は会話履歴からプロンプトを再構成 | Agent 個別メモリを持たないため再起動後も JSONL から文脈復元可。長い会話は `max_history` で打ち切り(要約・ベクトル検索は今後) |
| 7 | Error Handling | 例外階層 + `ERROR` メッセージ両建て。ハンドラ例外はループを殺さず送信元に通知 | `request()` 中の相手側エラーは `RemoteAgentError` で即時失敗(timeout 待ちを回避) |
| 8 | Authentication | in-process では `RegistryAuthenticator`(登録済 sender のみ)。`TokenAuthenticator` を同梱 | 同一プロセスでは主に「誤 sender」防止。ネットワーク化時は Token / JWT / mTLS 実装を `Authenticator` Protocol に差し込む |
| 9 | Concurrency | Agent ごとに 1 受信タスク、`request()` は Future ベース、Manager は `gather` で並列委譲 | 1 Agent 内は逐次処理(順序保証・LLM レート制限に安全)。並列化したい場合は Agent を複数起動するか `handle` 内で task 化 |
| 10 | Testing | MockLLM + HTTP モックで完全オフライン、async テスト | 実 API との結合テストは含めていない(コスト・鍵の都合)。examples で手動確認可能 |

### 既知の制約

- **単一プロセス限定**(現時点)。`InMemoryTransport` はプロセスを越えられない。
- **at-most-once 配送**。Agent がクラッシュすると mailbox 内メッセージは失われる(履歴には残る)。
- **broadcast への返信集約は未実装**。個々の返信は履歴 `replies_to()` で追跡可能。
- **1 role に複数 Agent がいる場合、Manager は先頭 1 体を選ぶ**(負荷分散なし)。
- `JsonlHistory` は追記のみで、大量履歴では起動時 replay が遅くなる。

---

## 今後の拡張案

1. **ネットワーク Transport** — `Transport` を実装するだけで差替え可能
   - `WebSocketTransport`(FastAPI/`websockets`)、`RedisTransport`(Streams)、NATS / RabbitMQ
   - Google **A2A protocol** 準拠の `A2ATransport`(`Message.to_dict()` がそのまま JSON ペイロード)
2. **永続化の強化** — `SQLiteHistory` / `PostgresHistory`(`HistoryStore` を実装)、ack ベースの at-least-once 配送
3. **Agent 側の高度化**
   - `DirectorAgent`(複数 Manager を統括)、動的プランニング(LLM が plan を生成)
   - role ごとの負荷分散・再試行・フェイルオーバー
   - 長期記憶(要約・ベクトル検索)を `HistoryStore` の上に構築
4. **LLM Adapter** — streaming、tool / function calling、レート制限・リトライ、コスト計測(`usage` は既に返却)
5. **運用性** — Web ダッシュボード(Agent 状態・会話グラフの可視化)、OpenTelemetry トレース、
   `message_id` / `task_id` をキーにした分散トレーシング
6. **セキュリティ** — JWT / mTLS `Authenticator`、role ベースの送信権限(誰が誰にどの `message_type` を送れるか)

---

## ライセンス

Apache License 2.0 — [LICENSE](LICENSE)
