图像/视频 + 问题
        ↓
Qwen2.5-VL 一次采样 8 个回答
        ↓
计算答案正确性、格式等奖励
        ↓
对相同回答进行三种 token 归因
        ↓
高熵 token ∪ 视觉敏感 token ∪ 时序敏感 token
        ↓
只有这些 token 进入 GRPO + KL 损失
        ↓
更新 Qwen2.5-VL 参数


