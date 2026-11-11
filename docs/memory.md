# 长期记忆

nanobot 把可编辑的记忆保存在 Markdown 中，用 SQLite 建立可重建的检索投影。结构化条目、每日摘要和人物记忆默认开启；远程 embedding 默认关闭。你可以在聊天中明确要求记住、纠正或遗忘，也可以检查文件、证据链和 Dream 审计。

记忆以 **agent workspace** 为范围，同一 workspace 的不同聊天共用。切换聊天或项目工作目录不等于隔离长期记忆；需要隔离时使用不同的 [实例与 workspace](multiple-instances.md)。

## 文件布局与职责

```text
workspace/
├── SOUL.md                       # 稳定行为约定
├── USER.md                       # core 偏好与用户资料
├── prompts/dream.md              # 可选的 workspace Dream 指南
├── memory/
│   ├── MEMORY.md                 # core 事实；只保留经常需要的内容
│   ├── ledger.md                 # 代码维护的事件账本与永久 tombstone
│   ├── history.jsonl             # compact 摘要及来源、时间、递增 cursor
│   ├── .cursor                   # history 的 cursor 计数
│   ├── .dream_cursor             # 已完成 Dream 的消费位置
│   ├── claims.sqlite3            # 可丢弃的条目、chunks、FTS、向量投影
│   ├── bank/
│   │   ├── preferences.md        # 非 core 偏好
│   │   ├── facts.md              # 事实、无项目的决定和承诺
│   │   ├── experiences.md        # 经历
│   │   └── circumstances.md      # 近况
│   ├── projects/<slug>.md        # 项目实体、进度、决定、待办
│   ├── people/<slug>.md          # 人物实体、别名、有证据的信息
│   └── daily/YYYY-MM-DD.md       # 每日摘要、无项目进度、取代/过期归档
└── dreams/YYYY-MM-DD.md          # 反思、待确认、代码生成的变更审计
```

原始聊天在 workspace 外的实例会话存储中，通常为 `sessions/<workspace-id>/sessions.sqlite3`。workspace 的 `.nanobot/workspace-id` 与会话目录中的 `.workspace` 标记用于定位对应存储；自定义实例配置会改变根目录。这个数据库与 `memory/claims.sqlite3` 的用途不同，不能把原始会话库当作可丢弃索引。

Markdown 和 `ledger.md` 是持久状态来源。每条 claim 的行尾含版本化注释，例如：

```markdown
- 早餐饮品只喝手冲咖啡 <!-- claim:c_0123456789ab {"v":1,"kind":"preference","status":"active","evidence":{"quote":"我只喝手冲咖啡"}} -->
```

代码写入的完整元数据还包括创建/更新时间、来源、到期时间、取代链接和人物/项目 ID。注入模型时会去掉机器注释。旧的 `<!-- claim:c_… -->` 仍可读取；旧行缺少的证据、时间或置信度保持未知，删库重建不会伪造这些值。保留新格式行及账本时，重建可以恢复其原有元数据。

手工编辑后，下一次读取、检索或构建记忆上下文会按文件内容哈希同步投影，不改写 Markdown。坏标记或坏 JSON 按普通文本索引；重复 claim marker 只有第一处作为 claim，其余作为普通文本，但同一 ID 已失效时所有副本都退出当前召回，不会因旧副本仍标为 active 而复活。相同文件内容的问题只告警一次。显式维护 `reconcile_files()` 才会清除被遗忘的恢复行、整理归档及删除已遗忘实体页；Dream 和记忆恢复会调用这一步。

## 数据怎样流动

```text
日常对话 → compact 压缩摘要 → history.jsonl + daily
                                   ↓
                            定时或手动 Dream
                                   ↓
                  dreams 反思 / 待确认 + 有证据的抽取
                                   ↓
                    claims → core / bank / projects / people
                                   ↓
                    文件哈希同步 → chunks / FTS / 向量缓存
                                   ↓
                    稳定 system 内容 + 当轮注入 / 主动检索
```

- **明确记住或纠正**：主 agent 当轮调用记忆工具，来源由运行时写入 `session_key`、`message_id`，actor 为 `user`。普通闲聊不需要每轮额外调用 LLM 抽取。
- **compact**：现有压缩流程生成摘要，带短原话及来源写入 history；开启 `dailyNotes` 时同时镜像到当天笔记，行尾保存 history cursor 和来源行摘要哈希。daily 会话摘要属于历史材料，默认不检索、不注入；使用 `include_history=true` 可查询并看到 `historical` 状态。旧版 daily 的「会话摘要」节也按历史材料处理。摘要是压缩后的证据，不等同于逐字聊天原文；只有独立保存的当前有效 claim 才按人物、项目和生命周期规则参与自动召回。
- **Dream**：定时任务与 `/dream` 共用协调器。每个 workspace 同一时刻只跑一个；每批最多读取 20 条未处理 history，为摘要和行生成 `[h:123]`、`[h:123:2]` 等 evidence handle。模型只能引用当前批次的来源，写入时再次检查来源没有被遗忘或修改。actor、run_id 和来源信息由运行时注入。
- **反思与抽取**：先记录反思，低证据推断放「待确认」；有证据的内容通过 `memory_remember` 写入，项目状态通过 `memory_project` 修改。人物抽取必须有来源中可核验的原话。下一次 Dream 最多带入约 1800 字符的近期待确认项，它们不算证据。
- **收尾维护**：代码按 run_id 写审计，记录相关事件、claim ID、简短正文与内容提交 SHA；已遗忘条目不写正文。成功完成后才推进 cursor。没有新 history 也会执行过期归档、tombstone 清理及限量向量补算，不为此调用 Dream LLM。等待 LLM 或 embedding API 时不持文件锁。

`dreams/` 不进入 system prompt、默认 search 或自动召回。它会进入记忆版本管理；其中命中的行也属于显式遗忘范围。

## 条目、实体和生命周期

### 存放位置

`core` 表示保存位置的意图，不是额外的一份正文。无人物/项目归属时，路由如下：

| kind | 默认位置 | `core=true` 时 |
|---|---|---|
| `preference` | `memory/bank/preferences.md` | `USER.md` |
| `fact` | `memory/bank/facts.md` | `memory/MEMORY.md` |
| `experience` | `memory/bank/experiences.md` | 仅在确需常驻时使用 core |
| `circumstance` | `memory/bank/circumstances.md` | 通常保留在 bank |
| `decision` | `memory/bank/facts.md` | 通常关联项目或保留在 bank |
| `commitment` | `memory/bank/facts.md` | 承诺仍从消息侧注入，无需设为 core |
| `progress` | 当天 `memory/daily/` | 通常关联项目或保留在 daily |

有项目归属时优先写项目页，否则有个人归属时写人物页。一个 claim 可以同时带 `person_id` 和 `project_id`，只存一份正文，人物召回通过关联 ID 查找。项目页以 `# 项目名` 开头，带稳定 entity ID 和 `active` / `completed` / `paused` 状态，包含「进度」「决定」「待办」节。人物页保存稳定 ID、显示名和显式别名。名称或 slug 改动不改变实体 ID；中文 slug 保留中文，去掉路径非法字符，重名追加后缀。

### 状态与时间

- `active`：当前有效条目。
- `superseded`：被新条目取代。支持多对多链接，旧 ID、原文和证据进入 daily 归档，默认检索和注入排除旧条目。
- `expired`：超过 `expires_at`，退出当前记忆；维护时归档到 daily 并保留 ID 和状态。
- `retracted`：显式遗忘留下无正文的状态记录。正文、quote 和向量不留在可查询的条目记录中。

`due_at` 是承诺的截止时间，`expires_at` 是记忆的失效时间。逾期但未失效的承诺仍会注入，并标记「已逾期」。不要用 `expires_at` 代替待办截止时间，否则到期后它会退出当前记忆。

内部时间使用 UTC；日期输入、daily/dreams 分桶和 history 的本地显示时间使用 agent 时区。history 时间仍为 `YYYY-MM-DD HH:MM`。日期型到期值按该时区当天结束处理。

### 聊天与工具

| 目标 | 工具与行为 |
|---|---|
| 「记住我只喝手冲咖啡」 | `memory_remember(text, kind, quote?, core?, …)` 保存原子条目 |
| 「以后改喝茶」 | 先 `memory_search` 找旧 ID，再用 `supersedes=[旧 ID]` 保存新条目 |
| 「蓝桥项目预演完成」 | `memory_remember(..., kind="progress", project="蓝桥")` |
| 「项目完成了」 | `memory_project(project="蓝桥", status="completed", rename?)`；完成/搁置后不自动注入，但仍可主动检索 |
| 「记住小林的偏好」 | `person="小林"`，可提供 `aliases`；需要 quote 或可核验的当轮来源。Dream 要求来源原话 |
| 「为什么记得这件事」 | `memory_explain(claim_id 或 query)` 查看证据、事件、文件、归属和双向取代链，最多 40 个节点 |
| 「忘掉这件事」 | `memory_forget(claim_ids 或 query 或 person/project)`；多目标先返回候选，确认后使用 ID 或 `all_matches=true` |

`memory_search(query, include_history=true)` 可以查询过期和取代归档，结果会标明状态。已遗忘条目不会因此重新出现；`memory_explain` 对它只返回「已遗忘」和时间。

关闭 `people.enabled` 后，人物参数会被拒绝，也不会自动注入人物资料；已有页面仍保留，显式检索仍可读取它们。这个开关不是删除或访问隔离措施。

<a id="save-and-check-a-preference"></a>

### 保存并核验一个偏好

明确说「记住：发布说明请用简短列表」，可由工具当轮保存。若这只是普通聊天中提到的偏好，先 `/compact` 再 `/dream`，分别等待完成，然后查看 `/dream-log`。默认非 core 偏好位于 `memory/bank/preferences.md`；只有明确设为 core 的偏好才进入 `USER.md`。不要只检查旧版的 USER/MEMORY 文件来判断是否保存成功。

调用 `memory_search` 查到正文与 ID，再用 `memory_explain` 核对来源；在同一实例的新话题里验证是否按需召回。模型口头说「记住了」不证明落盘，Dream 也可能判断没有需要保存的长期内容。纠正时用旧 ID 建立取代关系；遗忘或恢复时按下文的确认和快照规则操作。

## 注入与检索策略

记忆的 system prompt 部分只放稳定 core：`USER.md` 和 `MEMORY.md` 的当前内容，过滤结构化承诺、进度、人物/项目关联条目、失效及被遗忘条目。`injectMaxChars=6000` 限制的是 `MEMORY.md`，不是整个 system prompt，也不限制 `USER.md`。截断提示是固定文字。查询变化不会把 bank、人物或项目资料拼进 system prompt；实际修改 core 后，system 内容才随之变化。

消息侧由一个 memory runtime context provider 统一添加，包含包装文字在内总预算约 **4500 字符**，按 ID 和正文去重，优先级为：

1. 未完成承诺：默认最多 5 条，逾期仍显示。
2. 活跃项目：最近更新的最多 3 个 active 项目，每个取最新几条进度/待办，总预算约 1200 字符。
3. 本轮提到的人：每人约 600 字符，总计约 1500。英文按词边界匹配，中文按完整名或显式别名匹配；单字、歧义名称不自动注入。
4. 查询召回：默认最多 5 条，从 core、bank、projects、people、daily 中召回未在 system 或前述部分显示的当前条目。先过滤状态、项目/人物范围和排除项，再限候选；短 `MEMORY.md` 也参与检索。

自动注入的每条路径（包括承诺和项目摘要）都会排除 completed/paused 项目和未提及的人物；所有路径共用每人 600、总计 1500 字符的人物预算。主动调用 `memory_search` 的范围更宽，适合查阅这些已保存资料。结构化记忆关闭后不提供这些工具或动态记忆块，回到原有文件记忆行为。

每轮构建 prompt 只做一次文件哈希同步，core 排除项、实体和动态召回复用同一读视图；承诺、最近项目和已提及人物使用按需、有上限的查询，不先加载全量 claim。主 agent 的准备工作在已有受限 worker 中执行，LLM 等待期间不持文件锁。词面搜索流式评分，仅保留有界候选，避免把全部命中载入 Python 后排序。

关键词索引使用 SQLite FTS5 trigram；两字中文通过同一 chunks 表的 `LIKE` 查询。只有 SQLite 不支持 FTS5 或 trigram 时才完全使用 `LIKE`，其他数据库错误不会被静默降级。中文按相邻字片段匹配，并过滤一组常见停用词，**没有真正的分词或语义理解**；共享字片段仍可能误召回，不同说法也可能漏召回。请结合来源、状态和 `memory_explain` 判断，不要把检索结果当作已核验的结论。

## 显式遗忘与恢复

遗忘先把 tombstone 写入 `memory/ledger.md`，再清理文件和投影。账本保存 ID、动作、actor、run_id、时间、来源或行摘要哈希，不保存正文与引文。即使后续文件写入失败，读取投影也会过滤已记录的遗忘目标。仅在会话副本中命中的关键词也会推进记忆代次，避免进行中的摘要或旧 provider 续接状态重新带回内容。

确认目标后，先基于同一文件快照求完整的目标及正文/quote 关联集合，再统一清理 USER.md、MEMORY.md、bank、projects、people、daily、dreams 中匹配的行（包括标题），以及 history 中匹配的摘要行。按 ID 遗忘也会使用该条正文和 quote 清理副本；新保存的 Dream claim 还通过来源 cursor/行哈希直接撤销原始来源，即使保存正文是改写也会清理对应 history 行和 daily 镜像。由同一来源行生成的关联 claim 一并撤销，旧 handle 不能再次保存；旧条目若只记录了整条 history 来源，会撤销该条来源供 Dream 使用的资格，避免旧 handle 再生成记忆；缺少行级元数据时不猜测或整条抹掉原摘要，只清理可定位的正文/quote 副本，原摘要中的其他信息仍保留。长度不足两个字符的内容不作为删除词，避免单字误删。history 只把命中行替换为 `[已遗忘]`，保留同一摘要里的其他要点。按人物/项目确认全部遗忘时，也会记录 entity tombstone 并删除实体页。成功结果只含 ID、位置、数量和状态，不把删除正文再次返回模型；正文预览只用于执行前的歧义确认。

保留账本时，恢复原标记行、恢复实体页或删除 `claims.sqlite3` 后重建，都不能让已记录的遗忘目标重新参与召回。`sync` 只过滤，Dream 或 `reconcile_files()` 才显式清掉恢复的文件内容。这个承诺针对已记录的 ID 和行摘要；它不是覆盖所有改写、同义表达或新来源的语义黑名单。手工删行、取代、过期也不等于永久遗忘。

### 遗忘 v1 已覆盖

- **受管记忆**：上述 Markdown、`memory/history.jsonl`、条目/来源关联闭包和 `claims.sqlite3`。按 ID 遗忘从原正文、quote 和已定位的来源行生成脱敏词；只有聊天副本命中时也继续清理，不因记忆索引没有命中而提前返回。
- **当前 workspace 的 `sessions.sqlite3`**：原始消息、工具正文和参数、分支/子代理会话、展示历史、压缩的 history graph、恢复检查点、待处理输入、metadata（含 `_last_summary`）、标题和预览。递归处理 JSON，命中的字符串整体替换为 `[已遗忘]`；工具参数 JSON 字符串先解码，保留字段结构、role/type/name 和 tool-call 等关联 ID。图中的名称、路径按正文处理。不命中的摘要保持原样，`last_archived` 不回退。
- **会话提交与摘要竞态**：工具把请求登记到 `SessionState`，当前轮（包括取消后的检查点恢复）提交成功后才清理会话；更新 revision/generation、展示历史字节统计和 owner 缓存，并清空该 workspace 所有持久会话的 provider state。旧代次草稿、检查点及展示历史不能回写。等待模型期间记忆代次变化时，archive 丢弃摘要及 raw fallback，不写入 history/daily。
- **SQLite 当前文件**：两库连接启用 `secure_delete=ON`；清理事务提交后，记忆索引重建 FTS 并提交，再在事务外执行 `VACUUM`；WAL 数据库随后执行 `wal_checkpoint(TRUNCATE)`。数据库忙或操作失败会报告未完成，不能视为物理清理成功。结构标识和关联 ID 有意保留，即使其中恰好包含关键词也不会改写。
- **已知本地文件副本**：workspace 内的 `memory/HISTORY.md`、`HISTORY.md.bak*` 和所选 home 下的 `history/cli_history` 按行清理；当前会话目录及 workspace 的旧 `*.jsonl`、会话目录的 `jsonl-backup/`、实例数据目录的 `webui/*.jsonl` 和 `webui/*.segments/*.jsonl` 解码后删除命中记录；对应 `.checkpoint.json` 整体解析并脱敏。gateway/api 日志及 `.1`–`.5` 轮转文件按行清理；本进程持有的 `RotatingTextOutput` 在写锁内清理并重开。CLI、WebUI 旧文件和日志属于 home/实例共享范围，匹配内容也可能来自其他 workspace。

工具返回中，`completed` 表示已完成清理的范围，`pending` 表示尚待当前轮提交后的会话清理，`incomplete` 表示失败范围，`uncovered` 表示未覆盖范围。不要把「已登记」解释为「全部清理完成」。未注入会话 owner 的独立工具调用会明确列出未清理的会话范围；直接调用 `MemoryStore.forget_memory()` 只负责记忆，不代替工具的副本清理。

**崩溃或失败属于部分完成**：v1 没有跨文件与两库的原子事务、持久化协调器、启动续做或维护屏障。进程崩溃、关闭时保留未完成轮次、I/O 错误或 SQLite busy 都可能留下尚未清理的副本；账本的召回屏障不等于副本已经清除。后续 Dream/维护可以清理账本能定位的受管记忆，但不会自动补做本轮会话和其他副本清理。正文已删除后，按 ID 重试不保证能重新得到原脱敏词、补齐全部副本；需使用仍然知道的原关键词重新发起遗忘并核对结果。没有跨进程写屏障，其他进程正在持有的日志或已发出的模型请求也不能保证不会在清理后再次写入。

### 版本恢复

`/dream-log` 查看代码生成的变更记录；`/dream-restore` 列出可恢复版本。`/dream-restore <sha>` 的含义是：**把受管文件集合恢复到该提交的父快照，即该次修改发生之前**，不是只反向修改该次 diff，也不是恢复到该 SHA 本身。父快照中不存在的受管页面会删除。

恢复集合包括 SOUL、USER、MEMORY、daily、bank、projects、people、dreams 和 Dream cursor，**不回滚 ledger**。恢复与 Dream 使用同一个 workspace 准入门；正在整理或恢复时，另一操作直接提示稍后再试，避免旧 Dream 覆盖恢复后的 cursor。恢复后立即执行文件清理、history 清理和投影同步，再记录恢复提交。归档已写成但源文件清理失败时，维护按 ID 的最终状态清理所有副本。`history.jsonl` 本身不在 Git 快照集合中。SQLite、WAL、SHM 不纳入记忆 Git。

GitStore 在独立 workspace 中初始化专用仓库，并为已有记忆仓库补充忽略规则；workspace 位于已有父级 Git 仓库内时不会再创建嵌套仓库。若你自行管理这些文件的版本，请同时管理恢复和备份策略。

### 遗忘不覆盖哪些地方

- **记忆 Git 历史**：被忘内容仍可在记忆的 Git 历史中找到。v1 不重建、不改写、不清理 GitStore 的过去提交、提交消息和对象。
- **WebUI 浏览器存储和当前屏幕**：本地服务端清理不会删除浏览器已有的缓存、已显示内容或其他客户端副本。当前轮的模型上下文也不是即时语义擦除；清理完成后重新打开会话，以服务端脱敏后的重放为准。
- **其他持久副本**：outbox、升级备份、导出/同步副本、tool-results 外置文件、附件、cron，以及其他 workspace 的会话数据库不在本轮扫描范围。
- **其他文件与系统日志**：SOUL.md、自定义 prompts、skills、任意项目文件、journald/docker 日志不在覆盖范围；上述 gateway/api 本地日志文件之外的日志需单独处理。
- **provider 服务端留存**：本地续接状态带有 workspace 记忆代次；遗忘后不再复用旧状态，同轮 runner 在下一次模型请求前重建 core 并移除旧 memory 来源块。已经发出的请求无法追回，provider 服务端的数据保留和删除仍由其管理。

需要更广范围的清理时，先停掉对应实例，核对 workspace、会话命名空间及上述未覆盖副本，分别在其所属系统中处理并保留 `ledger.md`。本地文件重写、`secure_delete`、`VACUUM` 和 WAL 截断不承诺磁盘取证级擦除；磁盘快照、备份介质及 provider 留存仍需单独处理。

## 配置与升级

WebUI 的记忆设置提供 Dream 整理、条目记忆、关系页和语义检索开关；按界面要求重启 gateway。频率、预算、embedding 模型等高级项直接编辑配置。旧 gateway 没有提供新字段时，客户端隐藏相应开关。

以下是 `agents.defaults` 下的默认配置片段，合并到已有配置即可：

```json
{
  "agents": {
    "defaults": {
      "memory": {
        "claimsEnabled": true,
        "dailyNotes": true,
        "people": { "enabled": true },
        "injectMaxChars": 6000,
        "recallLimit": 5,
        "commitmentLimit": 5,
        "embedding": {
          "enabled": false,
          "provider": "openai",
          "model": "text-embedding-3-small",
          "dimensions": null,
          "batchSize": 32
        }
      },
      "dream": { "enabled": true, "intervalH": 2, "modelOverride": null }
    }
  }
}
```

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `memory.claimsEnabled` | `true` | 结构化记忆工具、条目投影及当轮动态注入 |
| `memory.dailyNotes` | `true` | compact 摘要镜像到 daily；关闭不删除旧笔记，也不禁止 progress/过期/取代归档写入 daily |
| `memory.people.enabled` | `true` | 人物参数和提及人物时的自动注入；依赖结构化记忆 |
| `memory.injectMaxChars` | `6000` | MEMORY.md 的 system 内容预算；`0` 表示不截断，claims 关闭时不应用此限制 |
| `memory.recallLimit` | `5` | 当轮查询召回条数，1–20 |
| `memory.commitmentLimit` | `5` | 当轮承诺条数，0–20；`0` 关闭专门的承诺列表，匹配项仍可能通过查询或项目摘要出现 |
| `memory.embedding.enabled` | `false` | 可选远程语义检索；关闭时零 embedding API 调用 |
| `memory.embedding.provider` | `openai` | 引用 `providers.<name>` 的连接配置，接口需支持 OpenAI-compatible embeddings |
| `memory.embedding.model` | `text-embedding-3-small` | 向量模型名 |
| `memory.embedding.dimensions` | `null` | 由服务决定维度；设为正整数时原样传给 API，模型必须支持 |
| `memory.embedding.batchSize` | `32` | 每次 Dream 最多补算的 claim 数，同时是请求批量，1–256 |
| `dream.enabled` | `true` | 定时 Dream；关闭不会删除记忆，仍可手动 `/dream` |
| `dream.intervalH` | `2` | 定时间隔小时数，至少 1；不增加额外抽取任务 |
| `dream.cron` | 未设置 | 兼容 cron 覆盖，优先于 `intervalH` |
| `dream.modelOverride` | `null` | 可选的 `modelPresets` 名称；不是裸模型 ID。未设置时使用主 agent 的 runtime |

### 老 workspace 升级

缺省字段现在开启 claims、daily 和 people，并把 MEMORY.md 预算设为 6000；配置里明确写出的 `false` 或 `0` 会保留，不会被默认值覆盖。读取已有 Markdown 不重排、不添加标记；第一次保存新条目时才追加相应节。已有未标记正文继续作为普通文本检索。旧索引版本 0/1/2 会重建为版本 3，以应用摘要历史状态和重复 marker 的生命周期规则，向量缓存可丢弃，不会因此修改 Markdown；旧 `HISTORY.md` 的既有迁移流程仍可能创建 journal 和备份。

回到原有文件记忆模式时，合并以下片段并重启。仅关闭 Dream 不等于关闭结构化记忆；仅关闭 claims 也不会停止单独开启的每日摘要镜像。

```json
{
  "agents": {
    "defaults": {
      "memory": {
        "claimsEnabled": false,
        "dailyNotes": false,
        "people": { "enabled": false },
        "embedding": { "enabled": false },
        "injectMaxChars": 0
      }
    }
  }
}
```

关闭不会删除文件或账本，但旧文件模式没有结构化 tombstone 的读取过滤。若要关闭前确保文件已完成遗忘清理，可先运行 `/dream` 完成维护；不要在关闭后重新拷入未清理的旧文件。重新开启后账本仍会参与过滤。若也要停止后台整理，另设 `dream.enabled=false`。

### 可选 embedding

先配置支持 embeddings 的 provider，再开启 `memory.embedding.enabled`。连接复用 `providers.<name>.apiKey`、`apiBase`、`extraHeaders`、`extraQuery` 和 `useProxy`；代理地址来自 `network.proxy`，不隐式使用环境代理。模型与维度由 `memory.embedding` 单独指定。缺 provider、缺 key 或无效连接地址/代理会明确报错，不会当作普通服务故障吞掉。WebUI 保存相关开关时复用纯配置校验，不发送网络请求；校验失败会拒绝整个设置 patch，防止开关保存成功却令 gateway 无法重启。

只有 active 且未过期的 claim 建向量，普通 Markdown 行仍走关键词检索。缓存键为 claim ID、正文哈希及 provider、规范化实际 endpoint、model、dimensions 和路由参数指纹（headers/query 只保存哈希），BLOB 使用归一化 float32。正文变更、遗忘、取代、过期会移除相应缓存；换 endpoint、模型、维度或路由参数后使用独立向量空间，由后续 Dream 逐批补算，不混用旧缓存。初始化或日常检索不会批量补算文档向量；手动和定时 Dream 收尾都会补算，包括无新 history 的轮次。

每次查询请求一次查询向量，在工作线程里流式计算余弦，使用有界堆保留候选；当前余弦阈值为 0.3，词面侧与向量侧都仅保留至多 `4 × limit` 候选，用等权 RRF（`k=60`）融合。API 错误、坏 JSON 或无效向量响应会告警并回到关键词检索，内部数据库错误仍抛出。开启后会把待补算 claim 正文和查询文本发送给该 provider，可能产生费用；关闭时不调用 embedding API。

## 1 核 1 GB 建议与基准

默认关键词方案不加载本地模型，不需要向量数据库或常驻检索服务。优先保持 embedding 关闭，避免把每轮召回变成额外网络请求。保持 core 简短，勿把全部 history 塞入 USER.md 或 MEMORY.md；默认 6000 字符只是 MEMORY.md 的预算，不是整个上下文上限。

需要语义检索时，可从服务支持的较低维度（例如 512）和 `batchSize=16` 或 32 开始。一次 `/dream` 不会补齐大库，重复运行或等待后续周期即可。向量扫描为线性复杂度，数据量和维度增长都会增加 CPU 时间；线程只避免阻塞事件循环，不会消除 CPU 成本。关闭 Dream 定时任务后，需要手动 `/dream` 才能补算；API 延迟与费用不在本地扫描基准内。

以下保留 D1 基线实测（当前版本已进一步改为有界词面候选和共享读取视图，数字不是本轮重新测得）：使用 Python 3.12.15、Linux x86_64、合成归一化正值向量，1 万条 × 512 维，top-5，预热一次后计时 5 次。它衡量实际 SQLite 流式扫描、有界候选选择与融合，不包含 API、Markdown 同步、上下文装配或真实模型效果，也不是对所有 1 核 1 GB 主机的性能保证。

| 指标 | 实测 |
|---|---|
| 查询耗时中位数 / 最大值 | 299.30 ms / 304.40 ms |
| 单次查询 Python 分配峰值（tracemalloc） | 23.42 KiB；不含 SQLite 的 C 层缓存 |
| 整个进程峰值 RSS（含数据准备） | 90068 KiB，约 88 MiB |
| 原始 float32 向量体积 | 19.53 MiB |
| 合成数据索引文件 | 51.25 MiB，含条目、元数据、FTS 与向量 |

从仓库根目录复现：

```sh
.venv/bin/python -m scripts.benchmark_memory_vectors --count 10000 --dimensions 512 --repeats 5 --limit 5
```

脚本只使用临时目录和合成数据，不调用 provider。输出 JSON 包含计时及 Python 分配峰值；进程 RSS 需用操作系统工具另行测量，不能把 tracemalloc 数字视作总内存。

## 效果自测

### 确定性场景：不调用 LLM

`tests/agent/test_memory_eval.py` 用「本地名称 + memory_remember 参数」表格描述对话事实，再通过真实记忆工具、MemoryStore、SQLite 和 ContextBuilder 检查结果。覆盖记住后召回、改口与双向证据链、过期归档、逾期承诺、跨文件/摘要/反思/向量遗忘、删库与快照恢复、人物证据及提及规则、项目完成退出注入。trigram 与 LIKE 两种路径都会运行。

固定问题集包含 12 个查询，分别统计 **recall@3**（正确 claim 是否进入前 3 条搜索结果）和当轮上下文命中率，两者断言均为 **至少 90%**。这是脚本化词面检索与生命周期回归，不代表自然语言语义召回率，也不衡量 LLM 是否正确抽取。JUnit 属性可保留两项指标：

```sh
.venv/bin/python -m pytest tests/agent/test_memory_eval.py -q -o junit_family=xunit1 --junitxml=/tmp/memory-eval.xml
```

### 可选真实 LLM 评测：不进入 CI

`scripts/memory_eval_llm.py` 读取现有 nanobot 配置中的主模型、Dream 模型和 provider 连接，使用四条合成对话事实执行真实对话、compact、统一 Dream 协调器和检索检查。只有运行该命令才会访问模型服务并产生费用；本脚本不使用第二个 LLM 来评分。

```sh
.venv/bin/python -m scripts.memory_eval_llm --config ~/.nanobot/config.json --output /tmp/nanobot-memory-eval-01
```

`--output` 必须是尚不存在的新目录。脚本不保存或打印 provider 配置，不读取现有 workspace 的记忆；评测 workspace、独立会话库和 `report.json` 都留在该目录供检查。用户配置文件不变。为比较记忆流程，评测副本使用默认启用的 claims/daily/people、关闭 embedding 与统一会话，并使用普通重试策略；主模型及 Dream 模型选择保留。对话和 compact 阶段不开放工具，Dream 使用生产协调器的受限工具集，可能在评测 workspace 中生成本地记忆 Git 版本。

报告检查对话收到回复、compact 非 RAW 降级、history/daily 生成、Dream 完成与 cursor 推进、摘要来源、人物原话、项目归属和人物按提及注入，并记录固定字面锚点的 claim recall@3，阈值为 **75%**。仅命中未结构化的 daily 摘要不算 claim 召回成功，注入检查也使用实际 claim ID。全部检查通过退出码为 `0`，正常完成但未达标为 `1`；配置或运行错误直接报错，不伪造成功报告。低分时先查看 `checks`、`probes` 和产物文件，区分抽取缺失、措辞变化和真正的检索问题。这是一个小规模端到端探针，不是模型排名或质量保证。

CI 只用脚本化 provider 测试这个评测入口，既验证成功路径，也验证漏抽人物信息会被报告为失败；不会调用真实 LLM。

## 定制 Dream

`/dream-prompt init` 在当前 workspace 创建 `prompts/dream.md`。其中有内容时作为该 workspace 的 Dream 指南，当前 history 及 evidence handle 仍由运行时追加；删除文件或置空即可恢复默认。将偏好写成简短规则，低证据推断继续放入「待确认」，不要要求模型绕过来源校验或把全部反思灌回 core。内置指南按 structured/legacy 模式渲染：结构化模式只通过记忆工具维护受管页面，普通过期/取代由代码归档，不能用永久遗忘代替日常清理；旧文件模式仍允许定点编辑。自定义指南若沿用了旧模板，应同步这些维护规则。
