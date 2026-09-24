# 集群 3 / B200 踩坑与修复索引

更新：2026-09-25。当前训练结果见 [结果报告](../reports/b200-ktr-full-20260924/RESULTS.md)；旧故障细节见 [流水线验证](archive/B200_PIPELINE_VALIDATION.md)。

目录整理后入口统一为仓库根`run.sh`，其余`.sh`在`scripts/`；旧H200/通用安装脚本以`.sh.txt`存于`docs/archive/scripts/`，不能当B200入口。55项相关检查通过（原43项+5项目录/链接/路由/归档追踪+7项交接同步）。仓库默认忽略txt，4个历史脚本原文已显式跟踪，避免只在本地归档却漏交付。用户baseline的exit130为主动停止，不计为新故障。

| 坑 | 证据/原因 | 已采用办法与边界 |
| --- | --- | --- |
| 两个集群不是同一块盘 | 控制端文件修改不会自动到运行端 | Git独立分支、bundle+SHA跨盘交付、全新release；禁止覆盖运行源码 |
| 有标注不等于媒体齐全 | 曾缺233个训练视频路径 | 补齐媒体路径后完整核验16916条；不删标注凑通过 |
| CPU解码超时被当成坏数据 | 原16条超时，重新单线程核验全部通过 | 每worker限制CPU线程，精确缓存，训练只读缓存；原日志阻塞为推断，不冒充已证明的唯一根因 |
| 全量解码与GPU串行等待 | 原预检耗时长且多次重复 | 热缓存完整检查约几十秒；DataLoader每rank2worker预取。冷缓存仍需先完成全量门禁，不宣称所有CPU预检与训练完全重叠 |
| 控制终端堵塞/断线 | 训练输出可能被管道背压阻塞 | detached生产进程写常规日志，独立tail查看；关闭查看器不停止训练 |
| 保活争抢GPU或退出后消失 | smoke/full转换及故障退出 | 同节点锁、官方controller、CPU阶段保活守护、GPU前暂停、释放后恢复；SIGKILL/节点故障无法保证EXIT trap执行 |
| 新release仍引用旧环境清单 | `environment manifest repo_root mismatch` | 实测重新生成release专属manifest，并显式传入；不绕过门禁、不伪造旧清单 |
| 2114步正常但末批崩溃 | gather_for_metrics按prompt remainder=4截断G8 completion reward | completion训练统计用gather；保留4个显式padding，loss/selector不变；KTR与baseline共用修复 |
| checkpoint存在但入口不会续跑 | 旧full未传resume参数 | 增加B200_RESUME_FROM_CHECKPOINT与数据/variant/分片门禁；2100恢复并真实完成15步 |
| 将checkpoint当新基座 | 会变成权重热启动并可能改变KL reference | 原base/reference不变，仅Trainer恢复完整状态；baseline明确清空resume参数 |
| 15步恢复的速度/平均loss异常漂亮 | Trainer混合累计2115步与局部耗时/损失 | 单独分析15条日志；原1–2100+恢复2101–2115去重；不报告5.235steps/s为真实吞吐 |
| smoke通过不代表epoch尾部通过 | 单步smoke没有触达数据尾部 | 增加尾批统计回归、恢复门禁测试，并以真实2115/2115作为本次尾部验收 |
| 把训练reward当模型评测结论 | 两组最后100步奖励接近，但无独立测试集结果 | 只作训练统计比较；另行固定评测集和生成参数，不以不同token掩码的loss排名 |
| 运行/文档时间与源码身份混淆 | UTC完成时间可对应次日北京时间；报告HEAD晚于运行commit | 同时保留完成UTC、报告日期、运行源码SHA；报告提交不改冻结release |
| 请求像素上限与实际视频上限不同 | Qwen原有video cap另有限制 | 分别记录请求401408和有效cap约105369；不悄悄抬高实际视频分辨率 |

新版43项相关测试通过（尾批3、恢复5、baseline入口1、full wrapper8、生命周期12、数据/runtime/smoke门禁14）。新版8卡paired smoke两者通过；真实KTR恢复、最终保存和保活恢复通过。正式baseline现已完成2115步，最终保存及保活恢复通过；独立质量评测仍待运行。见[配对报告](../reports/b200-paired-20260925/COMPARISON.md)。
