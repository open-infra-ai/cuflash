# Agent Note: 独立 kernel 作品，不接入旗舰运行时路径

Status: implemented

## Problem

cuflash 的 FlashAttention 前向/反向与 FlashDecoding 可用于真实推理路径；
一旦接入 tiny-llm，本仓就从「可独立审计的 kernel 深度作品」变成旗舰
运行时的依赖，两个仓的正确性声明互相纠缠，各自边界失守。

## Decision

cuflash 是独立 CUDA kernel 作品：FlashAttention 前后向（FP32/FP16/BF16，
FP16/BF16 前向接 WMMA）、FlashDecoding/Split-KV、Roofline 分析，
sm_70–sm_90。**不接入** tiny-llm generate 路径（对称约束记于工作区
`AGENTS.md` 规则 5 与 tiny-llm 侧笔记）。公共面冻结：CUDA API 表面与
支持的 `head_dim` ∈ {32, 64, 128}（见 CONTRIBUTING Scope Policy）。

## Alternatives considered

- **接入 tiny-llm 换真实负载验证** — kernel 直接获得旗舰场景背书最强；
  但本仓的性能叙事会被旗舰路径的约束（量化、分页、调度）稀释，且任何
  kernel 回归会拖垮另一个仓的 CI。
- **做成通用 attention 库发布** — 受众更宽；但维护面爆炸，与
  「教学可读性为底色、优化迭代叙事为定位」冲突。

## Consequences

- **收益**：性能与正确性声明只绑定本仓 benchmark 与测试，审计面闭合。
- **代价**：kernel 没有经旗舰路径的真实负载背书——这是刻意的取舍，
  由 tiny-llm 侧自包含路径补足端到端叙事。

## Verification

构建与测试中无 tiny-llm 依赖；README/ROADMAP 声明独立定位；
工作区 meta README 架构图标注「cuflash 证明 kernel 深度，但不是旗舰
请求路径的依赖」。
