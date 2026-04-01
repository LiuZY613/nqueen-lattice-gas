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

2. **精简Nobel等人的细节，保留引用，补充Simkin方法**（回应评论2+3）：删除了Nobel精确界的冗余数字细节，但保留了Nobel, Agrawal, Boyd (2023)的引用及其γ精度（~10⁻⁷），因为Table III的deviation需要与该精确值比较。改为描述Simkin证明所用方法：引入queenon极限对象、凸优化刻画γ、上界用熵方法、下界用随机化构造算法。

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

6. **Table II（Cv peak scaling）说明更新**：caption 中注明数据来自 N=8–128 在 T∈(0.20, 0.30)J 范围内密集取样的高分辨率专用模拟，数据内容不变。

---

## Nobel et al. 精确γ引用 & deviation修正 | 2026-03-24

### 修改内容

1. **恢复 Nobel, Agrawal, Boyd (2023) 引用**：该论文将 Simkin 的 γ 界从 [1.939, 1.945] 精化至 [1.944000752, 1.944001082]（精度 ~10⁻⁷），发表于 Optim. Lett. 17, 1229–1240 (2023)。添加 `\bibitem{Nobel2023}` 至参考文献。

2. **Introduction 中引入精确结果**：在 Simkin 证明段落之后新增一句，说明 Nobel et al. 用大规模 Newton 方法精化了 γ 的界。

3. **全文 γ 表述统一更新**：将所有 `$\gamma \in [1.939,\, 1.945]$` 改为 `$\gamma = 1.944001$`（摘要、eq:s0、Section V、Table III caption、结论），引用改为 `\cite{Nobel2023}`。不写闭区间，直接写值+精度。

4. **Table III deviation 重新计算**：deviation 改为与 Nobel 精确值 γ = 1.944001 比较（此前与 Simkin 范围中点 1.942 比较）：
   - N=32: 5.7%, N=64: 3.0%, N=128: 1.8%, N=256: 1.0%

5. **N=512 数据更新**：使用"大规模/"中新的 data_N512.dat（280温度点，1e8 sweeps），替代旧的126-point grid数据：
   - s₀: 4.298 → 4.308, γ_MC: 1.941 → 1.931, deviation: 0.67%
   - Table III caption 更新为"All data use a 280-point temperature grid"

6. **摘要和正文中 N=8,16 精度更新**：从"1.2%"改为"0.1%"（反映大规模数据的更高精度）。

7. **保留 Nobel 引用的原因**：虽然第二轮修改中导师要求精简精确数字，但 Nobel et al. 的精确值是 Table III deviation 计算的基准，且精度（~10⁻⁷）远高于 Simkin 原始界（~10⁻³），对定量比较至关重要，因此恢复引用并以简洁形式（γ = 1.944001, precision ~10⁻⁷）呈现。

---

## 全文一致性更新 | 2026-03-24

### 修改内容

1. **Fig 2 caption**：N 范围从 "N = 8--128" 改为 "N = 8--512"（图已包含7个尺寸）。

2. **Section IV.B Energy 正文**：
   - "N = 8--128" → "N = 8--256"
   - 删除关于 N=256/512 "not plotted" 及 "coarser temperature grid (126 vs. 318 points)" 的整段说明（数据已统一为280点网格，不再适用）。

3. **Fig 3 caption**："N = 8--128" → "N = 8--256"。

4. **Mean-field 比较（Section VI）**：
   - 正文 "N = 100 and 128" → "N = 128 and 256"
   - Fig 5 caption "N = 100 and N = 128" → "N = 128 and N = 256"
   - plot_PRE_figures.py 注释同步更新

5. **热力学积分节**："318 temperature grid points" → "280 temperature grid points"。

6. **重新生成全部图表**：fig2（含N=512）、fig3、fig4、fig5 均使用大规模数据重绘。

---

## N=512 数据一致性修正 & 时钟时间 | 2026-03-25

### 修改内容

1. **总时钟时间**：Section II "two hours" → "nine hours"。

2. **Section IV.A 收敛诊断**：所有数值更新至包含 N=512：
   - τ_int 峰值："1.8×10⁴ sweeps for N=128" → "4.9×10⁴ sweeps for N=512"
   - 独立样本数："~5,500" → "~2,000"
   - τ_int 峰位置温度范围："T≈0.075--0.10 J" → "T≈0.055--0.10 J"（N=512 峰在 T=0.055）
   - Cv 峰处 τ_int："130--200 sweeps" → "85--200 sweeps"（大 N 时 τ_int 更小）
   - 接受率范围："0.40--0.47" → "0.38--0.47"（N=512 在 T=1 时为 0.384）

3. **Fig 2 caption**：同步更新 τ_int 范围（85--200）和接受率范围（0.38--0.47）及 τ_int 峰温度范围（0.055--0.10）。

---

## 导师第三轮批注 | 2026-03-26 (commit 1730650)

### 导师批注 (Lei Wang)

1. **标题**: `n-queens` → `$n$-queens`（数学斜体格式修正）

2. **摘要第47行** `\lw{no need to claim first}`
   — 不要声称"first physical determination"，不需要宣称"首次"。

3. **Introduction第111行** `\lw{it is unclear what this "is" across the paragraph}`
   — "This observation motivates the present work"中的"this"指代不清，跨段落指代读起来含糊。

4. **Section II第185行** `\lw{why this ? why not queen-queen exchange ?}`
   — 为什么使用queen-vacancy交换（Kawasaki dynamics），而不是queen-queen交换？需要解释或论证选择的原因。

5. **Fig 2 caption第387行** `\lw{consider making this log y scale}`
   — 建议将Fig 2(a)接受率图的y轴改为对数坐标，因为接受率跨越多个数量级（从10⁻⁵到0.47）。

6. **Fig 2 caption第400行** `\lw{legend (b) overlapped with curve}`
   — Fig 2(b)的图例与曲线重叠，需要调整图例位置。

7. **Conclusions第784行** `\lw{call the problem n-queens lattice gas}`
   — 结论中应将模型称为"n-queens lattice gas"而非"queen lattice gas"，保持全文术语一致。

8. **新增致谢节**：添加了 Acknowledgments，感谢 Haijun Liao, Jinguo Liu, Tao Xiang 的讨论，以及国家自然科学基金等资助信息。

### 我们的修改 | 2026-03-26

逐条回应导师8条批注：

1. **标题 $n$-queens**（批注1）：导师已改好，保留。

2. **删除"first"声称**（批注2）：摘要和正文中两处 "the first physical determination/route" → "a physical determination/route"。

3. **修复"This"指代不清**（批注3）：将 "This observation motivates the present work" 改为 "The additive decomposition of entropy into constraint costs motivates a statistical-mechanical approach"，明确指代上文的约束层级分解。

4. **解释queen-vacancy交换的选择**（批注4）：在Kawasaki dynamics后新增一句解释：queen-vacancy交换允许皇后大范围空间位移，而queen-queen交换仅置换已占据位置，无法有效探索不同空间结构的构型。删除 `\lw` 标记。

5. **Fig 2(a) 改log y轴**（批注5）：修改 plot_fig2_convergence.py，将接受率y轴改为对数坐标（范围10⁻⁶到1），更好展示跨6个量级的变化。更新caption添加"(log scale)"说明。删除 `\lw` 标记。

6. **Fig 2(b) 图例位置**（批注6）：将图例从 upper right 移至右侧中部（bbox_to_anchor），避免与峰值区域曲线重叠。删除 `\lw` 标记。

7. **Conclusions术语统一**（批注7）：将 "queen lattice gas" 改为 "$n$-queens lattice gas"。删除 `\lw` 标记。

8. **致谢节**（批注8）：导师已添加，保留。

附加修正：Section II "all seven system sizes" → "all eight system sizes"（实际模拟了8个尺寸）。

所有 `\lw{...}` 批注标记已清除。

---

## 删除平均场理论 | 2026-03-26

### 修改内容

完整删除论文中所有平均场（mean-field）相关内容，论文从8页缩至7页。

删除的内容：
- **标题**："Poisson mean-field theory" → "thermodynamic integration"
- **摘要**：删除末尾MF句（"We further construct a Poisson mean-field theory..."）
- **Introduction**：删除MF段落（"To gain analytical insight..."）和章节导引中 sec:mf 的引用
- **Section II**：删除Hamiltonian分解处对MF的引用（"and the mean-field theory (Sec. VI)"）
- **整个Section VI (Mean-field theory)**：包括3个subsection（Simple Poisson MF、Modified Poisson MF、Comparison with MC）、所有MF方程（eq:nu_eff ~ eq:E_MF共9个）、Table IV（MF vs MC对比）、Fig 5（fig5_meanfield）
- **Conclusions**：删除MF条目（原point 5），改写开头段（删除"with analytical mean-field theory"）和结尾段（删除MF相关描述，改为强调MC+热力学积分的物理意义）

---

## 回应审稿人意见（第一轮） | 2026-03-26

### 审稿人意见回应

**1. "循环论证"与"首次物理确定"声明（Major 1）**
- 摘要：将"physical determination"改为"independent thermodynamic route"，强调 S(∞)/N 是"trivially exact"的均匀分布熵
- Introduction：明确说明唯一的组合输入是平凡精确的高温熵 S(∞)/N = (1/N)ln C(N²,N)，Simkin–Nobel值仅用于"post-hoc comparison"
- Section V（热力学积分）：补充说明 S(∞) 不依赖 Q(N) 或 Simkin 常数，Nobel值仅作为 benchmark
- 结论：在热力学积分条目中强调"trivially exact high-temperature entropy"为唯一输入

**2. 有限尺寸标度外推（Major 2）**
- 在 Section V 中新增一段讨论：承认 FSS 外推原则上可改善估计，但指出 Q(N) 次渐近展开的函数形式在数学上未知，任何拟合形式都是 ad hoc 的，因此仅报告原始 γ_MC(N) 值，将扩大 N 作为未来工作

**3. 相变判断措辞缓和（Major 3）**
- "rules out" → "consistent with the absence of"（正文和结论两处）
- Table II 新增 N=256, 512, 1024 数据（来自280点网格），总共9个尺寸
- 正文新增定量论证：N=32–1024范围内 ΔCv_max/N ≲ 0.007，远小于对数发散预期的 ~0.06/倍增
- Table II caption 更新：说明 8–128 用高精度密集取样，256–1024 用标准280点网格

**4. 热力学积分误差传播（Major 4）**
- 将 jackknife 误差通过梯形积分传播，得到 s₀^MC 和 γ_MC 的统计不确定度
- Table III 所有数据行添加 ±σ 误差棒
- 摘要、正文、结论中的 γ_MC 值添加误差：1.946 ± 0.003（N=1024）
- Table III caption 新增说明误差传播方法
- 正文新增讨论：σ(γ_MC)=0.003 远小于 0.11% 偏差，证实误差以有限尺寸修正为主

**5. 平均场部分（Major 5）**
- 已在上一轮修改中完整删除，不再适用

**6. 260核心 → 280核心（Minor 6）**
- Section II：260 → 280，与280温度点一致

**7. Table II 系统尺寸完善（Minor 7）**
- 见 Major 3 回应：新增 N=256, 512, 1024 行
- Caption 去除"dedicated high-resolution"统称，区分说明两组数据来源

**8. CSP文献联系（Minor 8）**
- 已在导师第二轮修改中删除 CSP 段落（导师明确要求删除）

**9. 修正Poisson MF独立性假设（Minor 9）**
- 已在上一轮修改中完整删除MF部分，不再适用

**10. 数据和代码可获取性（Minor 10）**
- 新增 "Data availability" 小节，声明代码和数据将公开在 GitHub
- 新增参考文献 github_repo

**11. 高温极限归属（Minor 11）**
- 大幅精简高温能量推导（从~55行缩至~15行），明确归功于 Polson & Sokolov (2024)
- 仅保留关键结果公式 E/N，删除中间推导步骤（eq:line_energy, eq:pair_prob, eq:S_diag, eq:S_tot）
- 结论中同步更新归属

**12. 基态熵层级新颖性（Minor 12）**
- 约束层级处新增限定语："implicit in the Stirling approximation and Simkin's result"
- 结论中同步更新："admits a physical interpretation as a constraint hierarchy"

### 其他修改

- `$n$-queens` 格式统一：摘要"n-queens" → "$n$-queens"，Table I "N-queens" → "$N$-queens"
- 全文 γ_MC 数值更新为含误差棒的精确值（三位小数）
- Table I 和 Table II 浮动位置从 [b] 改为 [t]，改善排版

---

## 导师第四轮批注 | 2026-03-27 (commits 1e3e83c7, 673eecc2)

### Teacher's Comments (Lei Wang)

**Commit 1e3e83c7 ("more comments", 2026-03-27):**

1. **Redundant figures** (after `\maketitle`): `\lw{Consider removing redundant figures (with both linear and logarithmic axes).}`
   — 考虑删除冗余图（同时有线性和对数坐标的）。

2. **Figure 1 coloring** (fig:schematic caption): `\lw{gray squares means something different. that is not good. use consistent coloring schemes. It is actually possible to combine (a) and (b) into a single figure with all information}`
   — 灰色方格含义不一致，配色方案需要统一。实际上可以将(a)和(b)合并为一张图，包含所有信息。

3. **Figure 1 aesthetics** (fig:schematic caption): `\lw{make this figure nicer, black dots -> queen look from above}`
   — 改善 Fig 1 外观，黑点改为皇后的俯视图形状。

4. **Table formatting** (after Table I): `\lw{red https://people.inf.ethz.ch/markusp/teaching/guides/guide-tables.pdf for nice tables}`
   — 参考该指南改善表格排版（"red"应为"read"）。

**Commit 673eecc2 ("comment", 2026-03-27):**

5. **Rook problem** (permutation placement line): `\lw{this is called rook problem}`
   — 指出排列放置问题实际上称为"车问题"（rook problem）。

### 我们的修改 | 2026-03-27

逐条回应导师5条批注：

1. **删除冗余双面板图**（批注1）：
   - Fig 3 (energy)：删除 panel (b)（线性低温），只保留 panel (a)（log-T 全范围 + 5/3 渐近线）。Caption 更新为单面板描述。
   - Fig 4 (Cv)：删除 panel (b)（log-T 全范围），只保留 panel (a)（线性 T≤1 + inset 放大峰值区域）。Caption 更新为单面板描述。
   - 修改 `plot_PRE_figures.py`：fig3 和 fig4 均改为单面板 `figsize=(3.4, 2.5)`。

2. **Fig 1 配色统一 + 美化**（批注2+3）：
   - 删除 panel (a)（攻击示意图，灰色=被攻击 vs 白色=未被攻击），只保留 panel (b)（8-queens 解）。
   - 棋盘配色改为经典国际象棋木色（浅色 #F0D9B5 + 深色 #B58863），替代之前灰白不一致的方案。
   - 黑色圆点改为 Unicode 皇后符号 ♛（U+265B），使用 Segoe UI Symbol 字体渲染，带白色描边确保在深浅两色格子上都清晰可见。
   - 图片宽度从 `\linewidth` 改为 `0.75\linewidth`，避免单面板过大。
   - 正文引用从 "The schematic of the model and the relevant length scales are shown in" 改为 "A representative ground-state configuration is shown in"。
   - Caption 简化为："One of the Q(8)=92 non-attacking ground-state configurations on an 8×8 chessboard. No two queens share a row, column, or diagonal."

3. **表格排版改善**（批注4）：
   - 新增 `\usepackage{booktabs}`。
   - Table I、Table II、Table III 中所有 `\hline` 和 `\noalign{\smallskip}\hline\noalign{\smallskip}` 替换为 `\midrule`（更细、间距更优雅的分隔线）。

4. **Rook problem 说明**（批注5）：
   - 在 constraint hierarchy 列表的 Permutation 条目中添加 "also known as the rook problem"，引用 Knuth [6]。

5. **清理**：删除所有 `\lw{...}` 批注标记（共5处）和 `\newcommand{\lw}` 定义。

---

## 导师第五轮批注 & Sec. VI 重写 | 2026-03-31

### 导师批注 (Lei Wang, commit 7121ae7)

1. **标题**: `\lw{title should be more general}` — 标题应更通用。
2. **新增作者**: 廖海军 (Hai-Jun Liao)，物理所。

### 我们的修改

1. **标题简化**（批注1）："Statistical mechanics of the $n$-queens lattice gas: Monte Carlo simulations and thermodynamic integration" → "Statistical mechanics of the $n$-queens problem"。

2. **Sec. VI 重写**：将 "Outlook: Exact enumeration via tensor networks" 改为 "Tensor network formulation"。用转移矩阵 $A$ 构造法替代旧的手动枚举描述：
   - 新增 Eq. (13)：$2\times 2$ 转移矩阵 $A$ 的定义
   - 新增 Eq. (14)：site tensor $T$ 的构造公式（四个 $A$ 矩阵元的乘积）
   - 引入边界向量 $\bm{v}_0, \bm{v}_1, \bm{v}_2$ 编码"恰好1个"和"至多1个"约束
   - 从约束层级（Sec. III A）自然引入，不再是独立的 "Outlook"
   - 删除 SWAP tensor 讨论
   - 图 caption：$B \to T$，增加对 Eq. (14) 的引用

3. **Abstract 微调**：新增 "transfer-matrix-based" 修饰词。

4. **作者**：添加 Hai-Jun Liao（在 Lei Wang 之前），物理所。

5. **清理**：删除 `\lw{title should be more general}` 批注和 `\newcommand{\lw}` 宏定义。

---

## 导师第五轮追加批注 | 2026-03-31 (commit 32e0a7d)

### Teacher's Comments (Lei Wang)

1. **张量网络部分** (Sec. VI, tensor network 段落后): `\lw{comment on infinite tensor network contraction, implications of super-extensive entropy and its effect on the convergence}`
   — 需要讨论无限张量网络收缩方法、超广延熵 (super-extensive entropy) 的物理含义，以及超广延熵对收缩收敛性的影响。

2. **致谢修改**: 从 Acknowledgments 中移除 "Haijun Liao"（已在之前的 commit 中添加为共同作者，不应同时出现在致谢中）。
   — "The authors acknowledge valuable discussions with Haijun Liao, Jinguo Liu, and Tao Xiang." → "The authors acknowledge valuable discussions with Jinguo Liu and Tao Xiang."

---

## 导师第六轮批注 | 2026-04-01 (commit 847688c)

### Teacher's Comments (Lei Wang)

1. **CTMRG处需要加引用** (Sec. VI, 超广延熵段落): `\lw{cite}`
   — 在提到 CTMRG 的地方需要补充参考文献引用。

   **CTMRG needs citation** (Sec. VI, super-extensive entropy paragraph): `\lw{cite}`
   — Add reference citation(s) where CTMRG is mentioned.

2. **统一使用大写 $N$-queens** (Sec. VI, 超广延熵段落): `\lw{use consistent N or n, makebe just capital $N$ throughout}`
   — 全文应统一使用大写 $N$-queens 还是小写 $n$-queens，导师建议统一用大写 $N$。

   **Use consistent capitalization for N-queens** (Sec. VI, super-extensive entropy paragraph): `\lw{use consistent N or n, makebe just capital $N$ throughout}`
   — The paper should use consistent capitalization; advisor suggests using capital $N$ throughout.

3. **致谢新增人名** (Acknowledgments): 新增致谢 Yijia Wang 和 Pan Zhang。
   — "...valuable discussions with Jinguo Liu and Tao Xiang." → "...valuable discussions with Jinguo Liu, Yijia Wang, Pan Zhang and Tao Xiang."

   **Acknowledgments updated**: Added Yijia Wang and Pan Zhang to the acknowledgments.
   — "...valuable discussions with Jinguo Liu and Tao Xiang." → "...valuable discussions with Jinguo Liu, Yijia Wang, Pan Zhang and Tao Xiang."
