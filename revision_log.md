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

---

## v2 第二轮修改 | 2026-03-24

### 导师评论 (Lei Wang)

1. **章节逻辑流** (Introduction前，第61行): `think about the logic flow of your sections, do they flow well??? III and IV does not seem to be in paralel to II and IV` — 重新审视各节之间的逻辑衔接，第III节和第IV节与第II节和第IV节似乎不平行
2. **γ精确值** (第87行): `why is is outsize 1.942 ?` — 质疑为什么Nobel等人的精确界 [1.944000,75, 1.944001,08] 在1.942之外
3. **精确数字的必要性** (第88行): `do you need to list these precise numbers here ? do they matter to your story` — 质疑是否需要列出这些精确数字，它们是否对论文叙事重要
4. **γ的意义** (第94行): `with unknown o(1) does gamma make sense ?` — 在o(1)项未知的情况下，γ是否有意义
5. **k_B=1的位置** (第125行): `move your kb = 1 remark here` — 将k_B=1的说明移到此处（即"We embed the N-queens problem..."之后）
6. **CSP段落** (第144行): `why this paragraph ?` — 质疑关于约束满足问题(CSP)段落的必要性

### 我们的修改

根据导师6条评论，进行了以下修改：

1. **合并第III节和第IV节**（回应评论1）：将"Ground-state entropy"和"High-temperature limit"合并为新的第III节"Analytical results"，分别作为子节A和B。更新了Introduction中的章节导引段落。

2. **删除Nobel等人的结果，补充Simkin方法**（回应评论2+3）：删除了Nobel, Agrawal, Boyd的精确界及其引用（Nobel2023）。改为描述Simkin证明所用方法：引入queenon极限对象、凸优化刻画γ、上界用熵方法、下界用随机化构造算法。

3. **移除所有o(1)项**（回应评论4）：
   - eq:QN: $(1 \pm o(1))N/e^\gamma$ → $\sim N/e^\gamma$
   - Introduction中 $s_0$ 表达式: 移除 $+\,o(1)$
   - eq:s0: 移除 $+\,o(1)$
   - 热力学积分节中对o(1)的引用也一并移除

4. **移动 $k_B = 1$ 说明**（回应评论5）：从独立段落移至"perform extensive Monte Carlo simulations"之后的括号说明中。

5. **删除CSP段落**（回应评论6）：删除了关于约束满足问题的段落及其引用 Mezard2009 和 Krzakala2007。

6. **γ范围修正**：全文将 $\gamma \approx 1.942$ 替换为 Simkin 给出的精确范围 $\gamma \in [1.939,\, 1.945]$。

7. **结论展望修正**：将未来工作的系统尺寸建议从 $N \gtrsim 256$ 改为 $N \gtrsim 512$。

8. **清理**：删除了所有导师 `\lw{...}` 评论标记。

---

## CPL 版本 | 2026-03-24

新增 `CPL/` 子目录，将论文改写为 Chinese Physics Letters (IOP `iopjournal` 格式) letter 版本，作为 PRE 长文的精简对照版。主要变化：

- 删除全部平均场理论内容（Section VI、Fig 5、Table II）
- 删除收敛诊断小节（Fig 2）
- 合并 Energy 和 C_v 图为一张并排双图（Fig 2）
- 保留核心结果：约束层级、高温精确极限、C_v 收敛与无相变、热力学积分提取 γ
- 参考文献从 11 条精简至 10 条
- 总页数：5 页（PRE 版约 9 页）

---

## 大规模数据替换 | 2026-03-24

使用"大规模"文件夹中的真实模拟数据（N=8,16,32,64,128,256，各280温度点，1e8 sweeps）替换 figures/ 中的旧数据，重新绘图并更新论文。

### 修改内容

1. **figures/ 数据替换**：删除全部旧 .dat 文件（含 data_N100, data_N512, data_N1024, data_L*_N* 等），复制"大规模/"中 data_N8~N256.dat 共6个文件。

2. **plot_fig2_convergence.py**：Ns 从 [8,16,32,64,100,128] 改为 [8,16,32,64,128,256]。

3. **plot_PRE_figures.py**：
   - 删除对外部 task2 目录（精确结果/密集结果/高温结果）的依赖，改为从本地 data_N{N}.dat 加载
   - Ns 改为 [8,16,32,64,128,256]
   - Fig 5 的 MC 对比从 N≥100 改为 N≥128

4. **重新生成** fig2_convergence, fig3_energy, fig4_cv, fig5_meanfield 的 PDF 和 PNG。

5. **Table III（热力学积分）更新**：
   - 删除 N=100 行（大规模数据中无此尺寸）
   - 新数据：N=8: s₀=0.566 (0.1%), N=16: 1.031 (0.1%), N=32: 1.633/γ=1.833 (5.6%), N=64: 2.274/γ=1.885 (2.9%), N=128: 2.943/γ=1.909 (1.7%), N=256: 3.620/γ=1.925 (0.9%)
   - N=512 行保持不变
   - 表格说明从"318-point grid; N=256,512用126-point"改为"280-point grid; N=512用126-point"

6. **Table II（Cv peak scaling）说明更新**：caption 中注明数据来自 N=8–128 的高分辨率专用模拟（dense temperature sampling near the peak region），数据内容不变。

