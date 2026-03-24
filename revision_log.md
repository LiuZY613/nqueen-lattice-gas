# Revision Log — eassy_nqueen

## v2 → v2 (revised) | 2026-03-24

### 导师修改 (Lei Wang, commit a6990ff)

1. **新增评论宏** `\lw{...}` 用于红色标注评论
2. **标题修改**: "queen lattice gas" → "n-queens lattice gas"
3. **摘要修改**: "We investigate the queen lattice gas" → "We investigate the n-queens problem as a lattice gas"
4. **评论**: `\lw{check all references avoid hulucination}` — 要求检查所有参考文献，避免幻觉

### 我们的修改

根据导师要求，逐一验证全部 14 条参考文献，发现并修正 4 处错误：

| 参考文献 | 原文（错误） | 修正后 |
|---------|------------|--------|
| Bezzel1848 | 页码 363 | **636** |
| Simkin2022 | Adv. Math. 410, 108735 (2022) | Adv. Math. **427**, 109127 (**2023**) |
| Bowtell2023 | Ann. Math. 197, 1 (2023) | **arXiv:2109.08083 (2021)**（未正式发表） |
| Luria2021 | "New bounds...", arXiv:2107.13460 | "**A lower bound for the n-queens problem**", arXiv:**2105.11431** |

其余 10 条参考文献验证无误：
- Nobel2023 ✓ | Knuth2022 ✓ | Zhang2009 ✓ | Polson2024 ✓ | Mezard2009 ✓
- Krzakala2007 ✓ | Kawasaki1966 ✓ | Metropolis1953 ✓ | Efron1982 ✓ | Binder2010 ✓

删除了导师的 `\lw{check all references avoid hulucination}` 评论标记。
