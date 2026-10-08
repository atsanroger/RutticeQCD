# DESIGN.md §7.6 多重網格（副本）

這是 `/home/chen/test/RUQCD/docs/DESIGN.md` 的 §7.6，2026-10-08 複製，為了讓別台機器上的 session 也讀得到
（`RUQCD` 不在 git 裡）。**正本仍是 RUQCD 那份**；兩邊不一致時以較新的為準，改了任何一邊請同步另一邊。
其他章節（§8 專案目錄等）只在正本裡。工作紀錄與量測在 `MDWF_MULTIGRID_LOG.md`。

### 7.6 多重網格

兩層的聚合式多重網格，當外層 Krylov 求解器的右預處理器。兩個參考實作，元件照它們寫，不另外發明：

- **clover：** Bridge++ `lib_alt` 的 `ASolver_MG`、`APrecond_MG`、`MultiGrid_Clover`、`AFopr_Clover_coarse`、
  `ASolver_SAP`、`ASolver_SAP_MINRES`；測試是 `test_alt_multigrid`。
- **domain-wall：** 使用者自己的 CUDA 版 `bridge_cuda/test_alt_amg_dw_GEMM`（`ASolver_MG_dw`、`APrecond_MG_dw`、
  `MultiGrid_Domainwall`、`src/lib_alt_Accel/Fopr/afopr_Domainwall_coarse-tmpl.h`、`block_fgmres_dw.h`）；
  參考 Kanamori–Chen–Matsufuru, PoS(LATTICE2024)414。

#### 7.6.0 原則：求解器與 fermion 無關

任何 fermion 都可以自由選擇求解器；多重網格只是其中一個預處理器選項，平滑器與粗網格解法也是選項。
fermion 只提供 §7.6.3 列的零件，不綁定任何求解器。可選的有：

| 角色 | 選項 | 限制 |
|---|---|---|
| 外層 | CG（正規方程）、BiCGStab、FGMRES、FBiCGStab；多個右手邊一起解時 block FGMRES | 預處理器不是固定的線性映射（多重網格、疊代式平滑器）時，只能用 flexible 的：FGMRES 或 FBiCGStab |
| 平滑器 | SAP（區塊內最小殘差）、GMRES(m)（固定步數）、MR 多項式（係數凍結）、CGNR | 作用在 V-cycle 的算符上（clover 是 D，domain-wall 是 A 或 D·C_PV⁻¹） |
| 粗網格解法 | BiCGStab、GMRES(m) | domain-wall 的粗網格用 BiCGStab 會出現 nan（CUDA 版的註解），預設 GMRES(m) |

參數檔與函式呼叫都是同一套選項；每個 fermion 的預設值寫在 §7.6.2，但不限定。

#### 7.6.1 兩者共用的骨架

| 元件 | 做法（兩個參考實作都一樣） |
|---|---|
| 聚合 | 4D 區塊；一個聚合包含區塊內所有 4D site 與所有 s（domain-wall 的第五維在 site 內層），依手徵分成兩半 |
| 手徵投影 | 用**算符自己的** γ5：P± v = (1 ± Γ5) v，Γ5 由算符提供（`mult_gm5`），不是寫死的逐 site 投影 |
| 測試向量的正交化 | 在每個聚合、每個手徵上做 Gram–Schmidt |
| 限制 R、延拓 P | coarse = 對每個聚合與手徵求 ⟨P± v_i, fine⟩；fine = Σ c P± v_i。Γ5 v_i 在 setup 後不變，快取起來 |
| 粗網格算符 | 8 個方向的最近鄰：對每個 test vector 與手徵，作用細網格算符的「區塊內部分」與「逐方向的 hopping 零件」，再投影到鄰近的聚合。粗網格的鄰居只有 8 個方向，因為建構時插入的因子（domain-wall 的 C_PV⁻¹）在 4D 是逐 site 的 |
| V-cycle | 一個 `LinearFn`：平滑、殘差、限制、粗網格解、延拓、修正。平滑器與粗網格解法是可換的成員 |
| setup | 初始測試向量 → 用平滑器解 → （選用）再作用一個算符 → 正交化 → 建粗網格算符；之後重複 n 次「以目前的 V-cycle 作用在測試向量上」 |
| 精度 | 預處理器單精度，外層雙精度（`Convert`） |

#### 7.6.2 兩個設定

| | clover（Bridge++ `test_alt_Multigrid.yaml`） | domain-wall（CUDA 版 `test_alt_Multigrid.yaml`） |
|---|---|---|
| 外層（預設，可換，§7.6.0） | `ASolver_FBiCGStab`（雙精度）；本專案目前只有 FGMRES，FBiCGStab 要補 | `ASolver_FBiCGStab` 作用在 A（dev6 版還註解著 Richardson、FGMRES）；多個右手邊時 block FGMRES |
| V-cycle 作用的算符 | D | A = (D_PV C_PV⁻¹)† C⁻¹ D |
| 初始猜測與平滑 | x = b，只做後平滑 | x = S(b)（前平滑），再做後平滑 |
| 平滑器（預設，可換，§7.6.0） | SAP：區塊依區塊座標著色，4 次，每個區塊 6 步最小殘差；區塊算符是 D 丟掉跨區塊的 hopping | 換過多次：`test_alt_dw_multigrid` 各版試過 `ASolver_SAP_dw`、`ASolver_SAP_dw2`（SAP 作用在 D·C_PV⁻¹，區塊內用 PV 解）、`ASolver_GMRES_m_Cmplx`，dev6 用 `ASolver_CGNR`；最新的 `test_alt_amg_dw_GEMM` 預設是批次的 block GMRES 作用在 A（固定 m 步，m = `smoother_number_of_iteration`，例 4，初始為零），`smoother_type: poly` 換成係數凍結的 MR 多項式（Horner 求值，不留 Krylov 基底） |
| 粗網格算符 | ⟨v_j \| D_區塊內 \| v_i⟩ 與 ⟨v_j \| hopping_μ \| v_i⟩ | 對每個 test vector、手徵、區塊奇偶：w = C_PV⁻¹（P± v_i，只留該奇偶的區塊）——C_PV⁻¹ 是精確的逐 site 求逆，不疊代——再取 ⟨P± v_j, D w⟩（同奇偶的區塊）與 ⟨P± v_j, hopping_μ w⟩（另一奇偶）。程式註解稱之為 A 的 Galerkin 粗網格 |
| 粗網格的預處理 | 無 | 用粗網格對角區塊的反矩陣（LU）左乘（`make_preconditioned`）；粗網格解之前先作用一次 |
| 粗網格解法（預設，可換，§7.6.0） | BiCGStab（`ASolver_BiCGStab_Cmplx`），\|r\|²/\|b\|² < 2e-3，最多 200 步 | dev6 用 BiCGStab；最新版用 GMRES(m)（註解：BiCGStab 會出現 nan），例 m = 5、\|r\|²/\|b\|² < 1e-2；多右手邊時 block GMRES |
| setup 的「再作用一個算符」 | 無 | 乘 D_PV：新的 test vector = D_PV · S(v) |
| 測試向量的來源 | 隨機向量 | 隨機向量，或特徵向量（參數 `testvector_type: eigenvector`，IRArnoldi_block）；例：8 個向量、setup 0 次 |
| 參數例 | 區塊 [8,4,4,4]（64×32³），16 個向量，setup 4 次 | 區塊 [3,3,3,3]（12⁴），Ls = 8，b = 1.5、c = 0.5，m = 0.005，m_PV = 1 |

#### 7.6.3 介面

程式只寫一份骨架，兩個設定是不同的成員與掛點：

- 細網格算符要提供：區塊內部分、8 個方向的 hopping 零件、Γ5（現有的 `HopTerms`：`mult_diag`、`hop_forward`、`hop_backward`、`mult_gamma5`，
  domain-wall 的 5D 算符同樣實作它，`ls = Ls` 在 site 內層）。
- 建粗網格算符時可插入一個逐 site 的右因子（domain-wall 的 C_PV⁻¹；clover 沒有）。
- 粗網格算符可選擇做對角區塊預處理。
- V-cycle 的成員是 `dyn LinearFn`：前平滑（可無）、後平滑、粗網格解法；V-cycle 作用的算符也是成員（clover 是 D，domain-wall 是 A）。
- setup 的平滑器、「再作用一個算符」（可無）、測試向量的來源（隨機或給定）是參數。
- 之後多加一層、改成 W 或 K cycle，都是換成員。

需要補的通用零件：GMRES(m)（固定步數可當平滑器、當粗網格解法）、FBiCGStab（照 Bridge++ `ASolver_FBiCGStab`）、
block FGMRES（用 `gram`、`axpy_mat`，§7.1）、MR 多項式平滑器、CGNR、
小型密集矩陣的 LU 與 Cholesky（`numerics`）、`Timed<F>`（包住任何 `LinearFn` 的計時）。
SAP 要能作用在任意「區塊內部分 + 逐方向零件」的算符上，包括 domain-wall 的 D·C_PV⁻¹（`ASolver_SAP_dw2` 的做法）。

#### 7.6.4 對照

多重網格是預處理器，正確性等於「外層收斂到真實殘差」，不需要逐位的參考值；要比的是迭代次數與時間：
clover 對 Bridge++ 的 `test_alt_multigrid`，domain-wall 對 CUDA 版在同樣參數下的紀錄（`run/` 裡的 log）。
測試向量來自各自的亂數產生器，所以迭代次數只能比到「同一個量級、隨參數的變化相同」；
要逐步比對時，兩邊都改用給定的測試向量（例如同一組特徵向量）。

#### 7.6.5 domain-wall 的修正：A′、Chebyshev 測試向量、Chebyshev smoother（2026-10-08；(a) 完成；(b)(c) 使用者同意、已實作（RutticeQCD `f72260f`），效果量測中；(d) 已量：不是截斷）

依據：`RutticeQCD/docs/MDWF_MULTIGRID_LOG.md` 的量測（12⁴ β = 5.7 與 16⁴ β = 6.0，Ls = 8，M5 = 1.0，m = 0.005）。
目標：外層用 BiCGStab 解正定算符，多重網格當固定的預處理器，不靠 FGMRES。

**(a) 目標算符改成論文的 A′。（使用者 2026-10-08 確認）**

| | 定義 | 右手邊與解 |
|---|---|---|
| 現在（`PvPreconditioned`，同 CUDA 與工作筆記） | A = (D_PV C_PV⁻¹)† C⁻¹ D | A x = (D_PV C_PV⁻¹)† C⁻¹ b |
| 改成（Kanamori–Chen–Matsufuru, PoS(LATTICE2024)414, Fig. 4 左） | A′ = (D_PV C_PV⁻¹)† D C⁻¹ | A′ y = (D_PV C_PV⁻¹)† b，x = C⁻¹ y |

理由：論文量的是 A′（條件數 1.96×10⁴）；A′ = X†Y（X = D_PV C_PV⁻¹、Y = D C⁻¹，同一種形式），m → M_PV 時是厄米正定的 X†X，A 不是。
量到的差別小：BiCGStab 不加預處理，5 個右手邊合計到 A 系統 1e-18 是 1587 對 1517 次（−4%），16⁴ 一個右手邊 −6%；
物理殘差 A′ 稍好。每次作用的成本相同（2 個 Wilson sweep、2 次 site 內求逆）。
介面：`PvPreconditioned` 的 `mult`／`mult_dag` 換成 A′，`precondition` 換成 X†，另加 `solution(x, y)`（x = C⁻¹ y）；
外層的使用方式從「解 A x = P b」變成「解 A′ y = X† b，再 x = C⁻¹ y」，所有呼叫端（tests、examples、`mdwf_multigrid`）一起改。
`mdwf_multigrid` 這一步只換 cycle 作用的算符（A → A′），粗網格的建法（`coarse_right` = C_PV⁻¹、test vector 乘 D_PV）先不動：
量測上它對 A 與 A′ 一樣沒有作用（平滑誤差都剩 0.707），換成 A′ 自然的 C⁻¹ 也一樣；要不要改由 (d) 決定。

**(b) 測試向量：Chebyshev 低通濾波。（已實作：`ChebyshevFilter`，`src/solver/chebyshev.rs`；尚未接進 `mdwf_multigrid` 的 setup）**

現在的測試向量是「隨機向量經一次 GMRES(4)」，平滑向量只有 7.7% 落在粗空間；反迭代到 1e-6 是 38%（8 個）與 47%（16 個）。
Boyle（arXiv:2409.03904）用 Chebyshev 低通濾波或 Lanczos 特徵向量，60–64 個。

- 濾的是厄米正定的 H = A′† A′（右奇異向量 = A′ 的右低模，也就是粗空間要表示的誤差方向）。
- φ = T_n(ℓ(H)) η，ℓ(x) = (2x − (hi + lo)) / (hi − lo)：[lo, hi] 映到 [−1, 1]（|T_n| ≤ 1，被壓住），x < lo 的模被放大約 cosh(n·acosh|ℓ(x)|) 倍。
  與 Bridge++ `AFopr_Chebyshev`（`degree_of_polynomial`、`threshold_value`、`upper_bound`，三項遞迴，三個工作向量）
  和 Grid `Chebyshev::InitLowPass` 同一種東西；本專案寫一個對 `LinearFn` 泛型的 `ChebyshevFilter`，與 fermion 無關。
- hi：power iteration 估 λ_max(H) 再乘 1.1；lo、n 是參數（起點：lo = hi × 10⁻³ 左右，n = 30–100，待量）。
- 一次濾波是 n 次 H = 2n 次 A′；n_vec 個向量放在同一個多向量場裡一起濾（§2，共用 link）。
- 濾完正規化，交給現有的 `Multigrid::with_parts(initial, …)`，setup 迭代 0 次。

**(c) smoother：Chebyshev 迭代（非厄米）。（已實作：`Chebyshev`、`MdwfSmoother::Chebyshev { steps, ratio }`）**

量測（12⁴、M5 = 1.0、5 個右手邊）：換成 Chebyshev smoother 後 BiCGStab + cycle 的迭代數從 70–122 變成 13–15（9 步、ratio 100），不再跳動；
但只用 Chebyshev 多項式當預處理器（不要 cycle）更便宜：17 步、ratio 300，12–13 次，≈ 420 次 A，與 FGMRES + GMRES(4) cycle 相當，仍輸 even-odd CGNE 約 2.2 倍。
下一步建議 even-odd 版的 A′（見 `RutticeQCD/docs/MDWF_MULTIGRID_LOG.md` 最後）。

**(e) even-odd 版：Ã = S_X† S_Y（2026-10-08 實驗，example `mdwf_evenodd_a`；尚未進函式庫，待寫設計與確認）。**
S_Y = 1 − D_eo C_o⁻¹ D_oe C_e⁻¹（Y = D C⁻¹ 的 Schur 補，S_D = S_Y C_e），S_X 同理於 D_PV。解 Ã y = S_X† b′_e、x_e = C_e⁻¹ y、x_o = C_o⁻¹(b_o − D_oe x_e)。
12⁴ M5 = 1.0：Ã 正定（隨機 +1.15、低模 +0.018），BiCGStab 不加預處理到物理殘差 4e-17 用 399 sweeps，**與 even-odd CGNE（391 sweeps、7e-17）打平**；全格子 A′ 約 1000。
之後的多重網格應建在 Ã 上。

凍結的 MR 多項式（`Polynomial`）放進 Krylov 會停滯（4 次 × 2 單獨當預處理器也停滯）；GMRES(m) 每次是不同的映射，BiCGStab 的迭代數因此在 45–119 跳動。
要的是固定、穩定、對正定非厄米算符有理論保證的多項式。

- Chebyshev 迭代（Manteuffel 1977；Saad, *Iterative Methods for Sparse Linear Systems* §12.1）：頻譜包在中心 θ、焦距 δ 的橢圓裡，
  x₀ = 0、r₀ = b、σ = θ/δ、ρ₀ = 1/σ、d₀ = r₀/θ；每步 x += d、r −= A′ d、ρ_new = 1/(2σ − ρ)、d = ρ_new ρ d + (2ρ_new/δ) r。
  固定步數就是 b 的固定多項式；δ 可以是複數（橢圓沿虛軸拉長時）。工作向量 x、r、d 三個，與 `Polynomial` 一樣。
- 當 smoother 用時只壓高模：橢圓包住 Re ∈ [hi/α, hi] 的部分（α 是參數，典型 10–30），hi 用 power iteration 估 A′ 的 |λ_max|，
  虛部寬度由 power iteration 的 Rayleigh quotient 給（第一版：δ 實數，橢圓退化成實數線段，之後再看虛部要不要納入）。
- 參考實作沒有非厄米的 Chebyshev（Bridge++ 與 Grid 都只有厄米、實數區間的版本），這一項是照教科書寫的，不是對齊。
- `MdwfSmoother` 加 `Chebyshev { steps, ratio }`；`Polynomial` 保留但不再是預設。

**(d) 粗網格算符：依量測決定，先不定。**

量到的：粗網格修正的程式正確（e = P c 修正後只剩 0.5–3%），但對平滑誤差 s 的修正離理論下限 √(1 − |PP†s|²) 很遠（16 個向量：0.990 對 0.883）。
兩個候選原因：A′ 是 2-hop（D 與 D_PV† 各 1-hop），探測截到 8 個最近鄰丟掉對角鄰居；或 A′ 非正規，Galerkin 是斜投影。
用「每次都真的算 P†A P」的精確粗網格算符分開（`examples/mdwf_coarse_check.rs`）：
- 若是截斷：`CoarseOp::probe` 改成保留 81 點（±1 的所有組合，Boyle 的做法；粗網格算符的儲存約 9 倍）。
- 若是斜投影：考慮 Kanamori 2023 設計筆記的結構（cycle 作用在 D，PV 只在粗網格 R D D_PV⁻¹ P），外層仍是 BiCGStab 解 A′。

量測結果（2026-10-08，8 個向量）：精確 P†A P 對 e = P c 剩 0.000、對 s 仍剩 0.995（探測版 0.033 / 0.995）→ **不是截斷**，81 點不必做。
推測是逐區塊投影 PP†s 在區塊邊界不連續、帶高模，Galerkin 修正因此不指向 s；候選：smoothed aggregation（P ← S P）、
或 Kanamori 2023 的結構。細節在 `RutticeQCD/docs/MDWF_MULTIGRID_LOG.md`。
之後：16 個向量精確版一樣（0.991）；smoothed aggregation P ← (1 − ωA)P 沒有用（0.993–0.995）；
用真正的低模 t（Chebyshev 濾波，|At|/|t| = 0.04）量，反迭代向量的精確修正也只剩 0.994。
更平滑的測試向量（反迭代、Chebyshev 濾波）讓探測粗網格變不準，cycle 變有害（BiCGStab 發散／NaN，FGMRES 43 → 78–98）。
局部一致性弱：2⁴ × 32 個向量也只抓到低模的 72%（3⁴ × 8：29%）。結論與替代方向（Kanamori 2023 結構、deflation、Chebyshev smoother）
見 `RutticeQCD/docs/MDWF_MULTIGRID_LOG.md`「結論與建議」。

**驗收：** 12⁴ 與 16⁴、M5 = 1.0、5 個右手邊，BiCGStab + 多重網格到物理殘差 |Dx − b|²/|b|² ≈ 1e-16，
「每個右手邊的細網格 Wilson sweep 總數」（含 smoother、setup 攤提另列）少於 even-odd CGNE（12⁴：約 400）。
