# H200 / B200 独立开发与交接协议

## 目录与分支

| 对象 | 源码位置（相对各自持久根） | 开发分支 | 职责 |
| --- | --- | --- | --- |
| 集群 1/2 的 H200 工作区 | `Video-KTR/` | `<owner>/multimodal-ktr-h200` | H200 配置、实验及报告 |
| 集群 1 的 B200 控制工作区 | `video-ktr-b200/repo/` | `<owner>/multimodal-ktr-b200` | B200 开发、联网 Git 操作、发布准备 |
| 集群 3 的 B200 运行工作区 | `video-ktr-b200/repo/`（当前运行旧目录） | 当前 run 的冻结记录 | 当前训练期间不更新 |
| 集群 3 的后续 B200 release | `video-ktr-b200/releases/<commit>/repo/` | 发布时的 B200 分支/commit | 以后从新目录启动，不覆盖旧 run |

前两个工作区是独立 clone，不共享 Git index / HEAD。集群 1 与 H200 GPU 共享持久盘；B200 运行盘独立，必须显式传输。旧 handoff 分支名已退役，其提交由两条实验分支保留。H200 新分支继承归档节点 `8fbaa46`，包含原主分支 `8a2e3cb` 的全部历史。只改开发分支名，不改项目路径、冻结release或运行身份。

## 文档的唯一责任人

- [handoff/STATUS.md](handoff/STATUS.md)：本分支唯一事实源，负责者更新配置、run、证据、问题、下一步。
- `handoff/peers/<另一profile>.md`：对方已提交状态的只读快照，记录来源 commit 和内容 SHA256；不是实时监控。
- `describe.md`、`DEVELOPMENT_PLAN.md`、`REMOTE_COLLAB_HANDOFF.md` 属于各自分支，不互相整篇覆盖；历史内容与 STATUS 冲突时，先核查 STATUS 的时间和运行证据。
- 跨分支共同修复按 commit 单独审阅/cherry-pick，再做目标 profile 的 smoke。不能为同步文档而 merge 整条实验分支。
- 聊天可自然交流；发给远程同事的正式交接记录按已有 `KT-HANDOFF/v1` 格式，附 profile、源码身份、run 和证据时间。

## 发布状态与接收对方状态

先在自己分支审阅并提交 `handoff/STATUS.md`，显式 push 自己的分支。对方只能看到已提交并传递的版本。

```bash
# 在自己的开发 checkout；将占位符替换为实际对方分支
git fetch origin
python3 tools/sync_handoff.py origin/<owner>/multimodal-ktr-<peer>
git diff -- handoff/peers/
# 首次生成时文件未跟踪，还需直接打开检查其内容
git add handoff/peers/<peer>.md
git commit -m "Import peer experiment handoff"
git push origin HEAD
```

工具只导入一个状态文件，不 checkout、不修改训练代码、不启动任务、不自动 commit/push。来源 profile 必须与自己不同；目标有未提交修改则拒绝覆盖。同一来源重跑不改变文件。

快照的 SOURCE_COMMIT 固定为对方最后一次修改 STATUS/角色的提交，而不是分支最新提交。对方仅提交了导入快照或其他无关文件时，不触发本地更新，避免两边“互相同步同步记录”的循环。

## B200 跨盘交付

开发端只从干净、已提交的 B200 checkout 导出完整 bundle：

```bash
git status --short
git rev-parse HEAD
git bundle create <持久传输目录>/b200-<commit>.bundle <B200开发分支>
sha256sum <持久传输目录>/b200-<commit>.bundle
```

通过已有 SSH 通道传到集群 3 的 `incoming/`。接收端先校验 bundle SHA，再 clone 到一个不存在的新 release 目录：

```bash
git clone --branch <B200开发分支> <incoming-bundle> <B200_ROOT>/releases/<commit>/repo
git -C <B200_ROOT>/releases/<commit>/repo rev-parse HEAD
git -C <B200_ROOT>/releases/<commit>/repo status --short
```

clone 只准备源码，不切换现有运行入口。使用 release 前必须确认原任务结束、source/smoke/runtime 门禁适用。手动启动时显式指定原项目持久根与机器私有配置，否则 release 的父目录不是实际 B200_ROOT：

```bash
B200_ROOT=<B200_ROOT> B200_ENV_FILE=<B200_ROOT>/b200.env \
  bash <B200_ROOT>/releases/<commit>/repo/launch_b200.sh ktr
```

baseline 同理替换最后一个参数。不要将 `b200.env`、venv、模型、数据或缓存放入源码 bundle。版本目录采用“发布后不编辑”的协作约定，不是文件系统强制只读。

当前正在运行的 B200 仍使用旧目录及其启动时的 commit；不能把新发布 branch/commit 写成当前 run 的版本。当前 H200 也保留启动时 dirty-worktree provenance，新分支不是对历史源码身份的追认。

## 结果回传与 GitHub

训练脚本只写集群本地结果，不会自动 push。结束后从该 run 提取到本分支的 `reports/<run-id>/`：

1. 固定运行配置、源码 commit/dirty patch 身份、模型及数据 manifest 哈希。
2. 脱敏后的时长/显存/利用率摘要、评测表、必要曲线与少量 CoT token 示例。
3. 大权重/checkpoint/完整日志所在存储的泛化标识和校验值，不提交其大文件。
4. 更新自己的 STATUS，提交并推送；另一边再导入其状态快照。

不要对源码根或数据目录做双向同步，不使用 `rsync --delete`，不强制添加被忽略的整个 artifacts。不能把 H200 与 B200 不同配置的结果直接当成 KTR/baseline 公平对照。
