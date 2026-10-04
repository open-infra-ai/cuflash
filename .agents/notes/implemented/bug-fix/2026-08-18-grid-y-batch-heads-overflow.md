# Agent Note: grid.y 65535 上限——batch×heads 拍平进 grid.x 维度

Status: implemented

## Problem

前向 kernel 用 `grid.y` 承载 `batch * heads`。CUDA 的 `gridDim.y` 上限是
65535，当 `B*H > 65535`（长序列大 batch 或多头配置）时 launch 越界失败
或行为未定义——大形状真实工作负载下直接不可用。

## Decision

把 `batch * heads` 从 `grid.y` 拍平移入 `grid.x` 维度（上限 2^31-1），
block 内还原 `(b, h)` 下标；同批加入 causal 路径的全未来块跳过优化
与 `B*H > 65535` 回归测试（commit `d144765`，2026-08-18）。

## Alternatives considered

- **文档声明 `B*H ≤ 65535` 限制** — 零改动；但限制在大形状场景必然被
  撞到，且失败模式是 launch 错误而非可读的参数校验。
- **改用 `grid.z`** — z 上限同样 65535，只是把墙挪了个方向。

## Consequences

- **收益**：`B*H` 上界从 65535 抬到 2^31 量级，大 batch 长序列可用；
  回归测试把边界形状钉住。
- **代价**：block→(b,h) 映射多一条除法/取模，launch 配置稍复杂。

## Verification

`B*H > 65535` 回归测试在测试套件中常驻（RTX 3060 Laptop 81/81 通过）；
修复提交 `d144765`。
