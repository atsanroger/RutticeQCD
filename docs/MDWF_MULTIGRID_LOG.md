# MDWF 多重網格：工作紀錄（2026-10-08）

給下一個 session 的交接。目標、量到的數字、結論、還沒做的事都在這裡；設計本身在
`/home/chen/test/RUQCD/docs/DESIGN.md` §7.6（不在 git 裡），副本是同目錄的 `DESIGN_7.6_multigrid.md`，
這份紀錄的修正方向在其中的 §7.6.5。

## 目標

用 **BiCGStab 而不是 FGMRES** 當外層，解 PV 預處理的正定算子 A，多重網格當預處理器。
參數以 **Ls = 8、M5 = 1.0** 為主（M5 = 1.8 次要），b = 1.5、c = 0.5、邊界 (1,1,1,-1)、m = 0.005。

## 測試組態（Bridge++ 產生的 quenched HMC，文字格式）

| 檔案 | 體積 | β | plaquette |
|---|---|---|---|
| `/home/chen/test/RUQCD/bench/hmc12/conf_quench_12x12x12x12.txt` | 12⁴ | 5.7 | 0.548 |
| `/home/chen/test/RUQCD/bench/hmc16/conf_quench_16x16x16x16.txt` | 16⁴ | 6.0 | 0.594 |

產生方式：`RUQCD/bench/hmc_setup.sh`、`hmc_two.sh`。

## 量測程式（`examples/`，都還沒 commit）

| 檔案 | 做什麼 |
|---|---|
| `mdwf_solvers.rs` | 各解法並排；每個解法跑 1e-10…1e-22 的「容許誤差階梯」，每階重算真實 \|Dx-b\|²/\|b\|²（A 系統的殘差和物理殘差差約 250 倍）。第 7 個參數只跑名字含該字串的解法 |
| `mdwf_precision.rs` | BiCGStab 解 A，f32 對 f64、隨機源對點源，看會不會 breakdown |
| `mdwf_smoother.rs` | 各候選預處理器 M 的 \|v-AMv\| 與 Re⟨v,AMv⟩（隨機向量與平滑向量） |
| `mdwf_testvectors.rs` | 隨機 vs 反迭代測試向量，配參考粗網格與 Galerkin，跑 BiCGStab／FGMRES |
| `mdwf_paper_a.rs` | 函式庫的 A 對論文的 A′；參數 `plain` 只比不加預處理、5 個右手邊 |
| `mdwf_coarse_check.rs` | 粗空間能表示多少低模、粗網格修正本身對不對；Galerkin vs Petrov-Galerkin，8 與 16 個向量 |

執行：`./target/release/examples/<name> CONFIG L M5 MASS [LS] [BLOCK] ...`（先 `cargo build --release --example <name>`）。

## 結論（12⁴、β = 5.7、M5 = 1.0、m = 0.005，一個組態；多數只有一個右手邊）

### 1. BiCGStab 可以直接解 A

- 不加預處理：1e-22 用 430 次，物理殘差 2.7e-20，平順、無 breakdown。
- f32 也不會 breakdown（M5 = 1.0：f32 119 次、f64 134 次到 1e-10；點源一樣）。M5 = 1.8 的 f32 只是停在 float 殘差地板（真實殘差剛好 1.0e-10），不是 breakdown。
- CUDA 程式註解「BiCGStab breaks down on the non-normal A」在 Ruttice 上不成立。Bridge++ CPU 的 `Solver_BiCGStab_Cmplx` 有 ω 修正（Omega_tolerance）與 breakdown 重起，**Ruttice 的 `BiCgStab` 兩者都已經有**（`56ddfec`，預設 ω 0.7）。CUDA 當時失敗的原因未查（可能是粗網格層、或當時 A 的 bug），使用者決定只測 Ruttice。

### 2. A 的定義和論文不同，論文的版本比較快

- 論文（Kanamori–Chen–Matsufuru, PoS(LATTICE2024)414, Fig. 4 左）：**A′ = (D_PV C_PV⁻¹)† D C⁻¹**（C⁻¹ 右預處理；解 A′y = (D_PV C_PV⁻¹)† b，x = C⁻¹ y）。
- CUDA 的 `mg_target_operator_PVprec.md`、工作筆記 `pvprec_amg.tex`、Ruttice 的 `PvPreconditioned`：**A = (D_PV C_PV⁻¹)† C⁻¹ D**（C⁻¹ 在左）。
- 論文的 A′ = X†Y（X = D_PV C_PV⁻¹、Y = D C⁻¹），m → M_PV 時是 X†X（厄米正定）；函式庫的不是。
- BiCGStab 不加預處理（右手邊 seed 91）：

| A 系統停在 | 函式庫 A | 論文 A′ |
|---|---|---|
| 1e-10 | 134 次，物理 2.1e-8 | 118 次，2.2e-8 |
| 1e-14 | 217 次，2.2e-12 | 196 次，2.0e-12 |
| 1e-18 | 361 次，1.9e-16 | 279 次，1.1e-16 |

- **更正：上表的 seed 91 剛好有利，換 5 個右手邊差距小很多**（`mdwf_paper_a ... plain`）：

| 12⁴，seed 91–95 合計 | 函式庫 A | 論文 A′ |
|---|---|---|
| A 系統停在 1e-14 | 1005 次 | 995 次（−1%） |
| A 系統停在 1e-18 | 1587 次 | 1517 次（−4%） |

  seed 92 反而是 A′ 慢（339 對 268）。16⁴ β = 6.0 只跑了 seed 91：1e-14 是 824 對 746、1e-18 是 1274 對 1202（−6%），
  其餘 4 個右手邊使用者決定不跑。物理殘差 A′ 稍好。**結論：A′ 稍好或持平，在 BiCGStab 的雜訊範圍內**；改用 A′ 的理由是它是論文的定義、結構上 m → M_PV 時是 X†X，不是速度。
- 改用 A′ 已寫進 DESIGN.md §7.6.5（草稿，待使用者確認）。
- A′ 對粗網格修正沒有幫助（見 4）。

### 3. 基準（物理殘差對迭代數）

| 解法 | 1e-16 附近的物理殘差 | 備註 |
|---|---|---|
| even-odd CGNE | 197 次，394 sweeps，5.7e-17 | 1e-22 是 282 次 |
| BiCGStab 解 A，無預處理 | 361 次（A 停 1e-18），1444 sweeps，1.9e-16 | A 的系統是全格子，比 CGNE 貴約 3 倍 sweeps |
| BiCGStab + cycle（GMRES(4) smoother） | 111 次（A 停 1e-16），2.3e-14；1e-22 是 157 次 | 迭代數會跳：同題 45–119 次（M 不固定） |
| FGMRES(20) + cycle（GMRES(4) smoother） | 43 次（A 停 1e-16），1.5e-14；1e-22 是 64 次 | 每次外層只作用 M 一次 |

cycle 的全部效果來自 smoother（cycle 和「兩次平滑」在三位數內相同），粗網格修正沒有作用。

### 4. 粗網格修正為什麼沒有作用

`mdwf_coarse_check`（P 由測試向量建，s = A x = v 的解、正規化）：

| 測試向量 | \|PP†s\|/\|s\| | e = P c 修正後剩 | e = s 修正後剩 | 理論下限 √(1-\|PP†s\|²) |
|---|---|---|---|---|
| 隨機 8 個 | 0.077 | 0.005 | 1.000 | 0.997 |
| 反迭代 1e-6、8 個（Galerkin） | 0.378 | 0.033 | 0.995 | 0.926 |
| 反迭代 1e-6、16 個（Galerkin） | 0.469 | 0.034 | 0.990 | 0.883 |
| 反迭代 1e-6、8 個（Petrov-Galerkin，Q 來自 A† 的反迭代） | 0.378 | 0.996 | 1.000 | 0.926 |

- **程式沒有錯**：誤差在 span(P) 裡就修得掉（e = P c 剩 0.5–3%）。
- **主因一：粗空間抓不到低模。** 隨機向量只有 7.7%；反迭代 41% 上下；向量加倍只到 47%。→ 測試向量的「品質」比數量重要，決定改用 **Chebyshev 低通濾波**。
- **主因二：實際修正離理論下限很遠**（0.990 對 0.883）。候選原因，還沒分開：(a) A 非正規，Galerkin 是斜投影；(b) A 是 2-hop（D 與 D_PV† 各 1-hop），粗網格截到最近鄰丟掉對角鄰居。下一步用「精確的 P†AP」（每次真的算 P†A P c，不截斷）把兩者分開。
- 我一度主張「A 不是 Γ5-厄米所以 P† 看不到殘差」，**已撤回**：用來支持它的指標 \|QQ†r\| 設計錯了（r = A s 就是原本的隨機向量）。Petrov-Galerkin 反而讓粗網格算子不能用。
- 試過、都沒改善的：參考粗網格 P†D C_PV⁻¹P、探測的 Galerkin P†AP、粗網格精確解、論文的 A′（配 P†D C⁻¹P 或 Galerkin）。所有變化在平滑向量上都留下約 0.707。

### 5. 凍結多項式 smoother 在 Krylov 裡停滯

- BiCGStab + cycle（多項式 4 次 smoother）：4000 次停在 3e-2；FGMRES(20) 同 cycle 停在 2.6e-4。
- BiCGStab + 「多項式 4 次 × 2」單獨當預處理器也停滯（40 分鐘未過 1e-10）；「8 次 × 2」可以收斂（216 次到 1e-10，比無預處理的 134 次還多）。
- 單獨作用一次時多項式是好的（\|v-AMv\| 與 GMRES(4) 相同、Re⟨v,AMv⟩ > 0），所以推測是擬合出來的 4 次多項式在某些低模上讓 λp(λ) 接近零或轉到左半平面。未驗證。
- 決定改用 **Chebyshev smoother**（見下）。

## 參考文獻與文件（本機）

- 論文：`/home/chen/red_gpu/ref/Spectrum_of_Preconditioned_Moebius_Domain-wall_Ope.pdf`
- Kanamori 2023 設計筆記：`/home/chen/red_gpu/ref/dwf_multigrid.pdf`。要點：外層、smoother、cycle 都作用在 **D_DW**；PV 只出現在粗網格（右預處理 R D (M_PV)_SAP P、修正 x += D_PV⁻¹ P x_c）；setup 是隨機 → SAP → 用 cycle 反覆改進（像 clover）；粗網格解法首選 BiCGStab。CUDA 版後來改成「全部作用在 A 上」（`bridge_cuda/test_alt_amg_dw_GEMM/asolver_MG_dw.cpp` 開頭的「Option 2」註解），原因是外層解 D 時 GMRES 發散、CGNR smoother 在 M0 = 1.0 發散、B_PV 近似產生負實部。
- CUDA 的 A 設計：`/home/chen/test/bridge_cuda/docs/mg_target_operator_PVprec.md`
- 工作筆記：`/home/chen/test/bridge_cuda/docs/latex/pvprec_amg.tex`（CUDA 的 12⁴ 設定用 `testvector_type: eigenvector`，40 個 Arnoldi 向量，setup 0 次；Ruttice 的 `MdwfMultigridParams::note()` 抄了 setup 0 次卻用隨機向量）
- Boyle 2024（arXiv:2409.03904）：60–64 個 basis 向量、區塊 4⁴ 或 4·4·6·4、Ls = 24、粗化的是厄米的 red-black 平方算子、粗網格保留完整 4-hop（81 點）、向量來自 Chebyshev 濾波或 Lanczos。

## 決定事項（使用者，2026-10-08）

- M5 = 1.0 為主。
- 測試向量改善品質：同意。
- 改用 Chebyshev：同意。
- 授權：問過 GPL；建議 GPL-3.0-or-later，**尚未決定、尚未加 LICENSE**。

## 計畫（2026-10-08 定）

設計細節在 DESIGN.md §7.6.5（副本：`docs/DESIGN_7.6_multigrid.md`）。每一步完成後更新本檔。

| 步 | 內容 | 狀態 | 估計 |
|---|---|---|---|
| 0 | 量測與診斷（上面 1–5） | 完成 | — |
| 1 | 精確 P†AP（`mdwf_coarse_check`）：分開「2-hop 截斷」與「斜投影」 | **完成（8 個向量）：截斷不是原因**，見下 | — |
| 2 | **A 改成論文的 A′**（§7.6.5 (a)，已確認）：`PvPreconditioned` 換成 A′ = X†Y、`right_hand_side`（X†b，原 `precondition`）、`solution`（x = C⁻¹y）；tests、`mdwf_multigrid` 文件、examples 一起改；粗網格建法先不動 | **完成，未 commit**；`stage3_mdwf` 13 個測試通過（含 Bridge++ 組態：`RUTTICE_BRIDGE_TESTS=/home/chen/test/RUQCD/bridge_ref`）；clippy、fmt 乾淨 | — |
| 3 | Chebyshev 測試向量（§7.6.5 (b)）：對 H = A′†A′ 低通濾波，`ChebyshevFilter` 放 `src/solver/` | **實作完成（`f72260f`）**；接進 `mdwf_multigrid` 的 setup 還沒做（目前在 example 裡用）；效果量測中 | — |
| 4 | Chebyshev smoother（§7.6.5 (c)）：非厄米 Chebyshev 迭代，`MdwfSmoother::Chebyshev` | **實作完成（`f72260f`）**；在 12⁴ 上的效果還沒量 | — |
| 5 | 粗網格算符（§7.6.5 (d)）：步驟 1 說不是截斷 → 「保留 81 點」不會有用；要查的是投影本身（見下的推測），或考慮 Kanamori 2023 結構（cycle 作用在 D） | 待決定 | 不確定（研究問題） |
| 6 | 驗收：12⁴、16⁴，M5 = 1.0，5 個右手邊，物理殘差 ≈ 1e-16，BiCGStab + MG 的細網格 sweep 總數 < even-odd CGNE（12⁴ 約 400） | — | 1–2 小時 |

成功標準：BiCGStab（不是 FGMRES）+ 多重網格，粗網格修正真的有作用（FGMRES 的外層數明顯低於「只做平滑」的 43、平滑誤差的修正接近 √(1-|PP†s|²)），且細網格 sweep 總數贏 CGNE。

## 步驟 1 的結果：精確 P†AP（12⁴，舊 A；A 與 A′ 在粗網格上表現相同）

| 反迭代 1e-6，Galerkin | e = P c 修正後剩 | e = s 修正後剩 | 理論下限 |
|---|---|---|---|
| 8 個向量，探測（截到 8 個最近鄰） | 0.033 | 0.995 | 0.926 |
| **8 個向量，精確 P†AP（不截斷）** | **0.000** | **0.995** | 0.926 |
| 16 個向量，探測 | 0.034 | 0.990 | 0.883 |
| **16 個向量，精確** | **0.000** | **0.991** | 0.883 |
| 8 個向量，Petrov-Galerkin（Q 來自 A† 反迭代），探測與精確 | 0.996 | 1.000 | 0.926 |

不截斷時 span(P) 裡的誤差完全修掉，平滑誤差卻一樣沒動 → **不是截斷**。
推測（未驗證）：s 落在 span(P) 的 38% 是逐區塊投影的 PP†s，在區塊邊界不連續、帶大量高模，A(PP†s) 不小，
Galerkin 修正因此不指向 s。對厄米算符這靠 smoother 或 smoothed aggregation（先平滑 P）補救；
但 cycle 加平滑也沒有改善（平滑誤差都剩約 0.707）。下一步可以試：
(1) 量 |A PP†s| 對 |A s|，確認上面的推測；(2) smoothed aggregation：P ← (1 − ωA) P 或 S P；
(3) Kanamori 2023 的結構。Petrov-Galerkin（Q 來自 A† 反迭代）讓粗網格算符本身壞掉（e = P c 剩 0.996），不是解法。

步驟 2 附帶的兩個數字（Bridge++ 4³×8 組態，`stage3_mdwf`）：BiCGStab 解 A′ 496 次（舊 A 443 次，**慢 12%**，
與 12⁴ 的 −4% 方向相反 → 速度不是改用 A′ 的理由）；BiCGStab + cycle（Ls 4、m 0.1）62–71 次（舊 A 356–488 次），
原因未確認，可能只是 BiCGStab 配變動預處理器的跳動。

## 之後的量測（2026-10-08 晚，使用者授權自行進行）

**smoothed aggregation 沒有用**（`mdwf_smoothed_aggregation`，8 個反迭代向量，精確 P_ω†A P_ω）：
P_ω = (1 − ωA)P，ω = 0、0.25、0.5、1.0 除以 |λ_max(A)| ≈ 4.53：平滑誤差 s 都剩 0.993–0.995。

**「區塊邊界粗糙」只部分成立**：|A v|/|v| 對 s 是 0.290，對 PP†s 是 0.528（粗 1.8 倍），對 s − PP†s 是 0.380，隨機向量 1.550。

**重要但書：s 不夠低。** s = A⁻¹v（v 隨機）的 |As|/|s| = 0.29，只比隨機向量低 5 倍；A 的最低模約 3e-3。
隨機向量在最低幾個模上的分量極小，A⁻¹ 放大的主要是大量中等模。所以上面所有「e = s 剩 0.99」量的是粗網格對中等模的效果，
不是對讓 Krylov 停滯的最低模。**用真正的低模重量**：Chebyshev 濾波 T_200(ℓ(A†A)) 作用在隨機向量上得到 t，|At|/|t| = 0.040
（`mdwf_lowmodes`）。反迭代 8 個向量：|PP†t| = 0.310，精確修正後 t 剩 0.994（下限 0.951）——對真正的低模一樣沒有作用。
**更好的測試向量讓 cycle 更糟**（`mdwf_lowmodes`，8 個向量、3⁴、GMRES(4) smoother、探測的 Galerkin 粗網格，到 A 系統 1e-16）：

| 測試向量 | \|PP†t\|/\|t\| | 精確修正後 t 剩 | BiCGStab + cycle | FGMRES(20) + cycle |
|---|---|---|---|---|
| 隨機（之前） | — | — | 111 | 43 |
| 反迭代 1e-6 | 0.310 | 0.994 | **發散**（3000 次，物理殘差 2e157） | 98 |
| Chebyshev 濾波（degree 100） | 0.294 | 0.992 | **NaN**（1501 次） | 78 |

推測：測試向量越平滑，截到最近鄰的探測粗網格算符越不準（e = P c 從剩 0.5% 變成 3–16%），粗網格修正注入錯誤方向；
FGMRES 能排除、BiCGStab 不行。所以「不是截斷」只對「單獨一次精確修正能消多少平滑誤差」成立；**放進 cycle 時，截斷的不精確會讓 cycle 有害**。
另外：BiCGStab 回傳「未收斂」但解已是 NaN，breakdown 檢查沒抓到（NaN 推測來自預處理器）——函式庫該在預處理器輸出非有限值時報 breakdown（待修）。

**局部一致性弱**（`mdwf_coherence`，Chebyshev 濾波向量，兩個低模誤差 t1、t2，|At|/|t| = 0.040、0.033）：

| 區塊 | 8 個向量 | 16 個 | 32 個 |
|---|---|---|---|
| 2⁴ | 0.43 / 0.45 | 0.57 / 0.59 | 0.72 / 0.74 |
| 3⁴（目前） | 0.29 / 0.31 | 0.40 / 0.42 | 0.53 / 0.56 |
| 4⁴ | 0.20 / 0.21 | 0.28 / 0.30 | 0.38 / 0.41 |
| 6⁴ | 0.10 / 0.12 | 0.15 / 0.17 | 0.21 / 0.24 |

（每格是 \|PP†t1\|/\|t1\| / \|PP†t2\|/\|t2\|；細網格維度 12⁴ × 8 × 12 ≈ 2.0M）

- 最好的 2⁴ × 32（粗網格維度約 83k，細網格的 4%）也只有 72%；Wilson MG 一般 ~1% 就抓到大部分。
- 同樣的粗網格維度，加向量比縮小區塊有效（3⁴×32：16k 維 0.53；2⁴×8：21k 維 0.43）。與 Boyle 用 60–64 個向量一致。

**結論與建議（給下一個 session）**

1. 「全部作用在 A 上」的聚合 MG，要讓粗網格修正有作用，需要：很多向量（≥ 32）、精確的粗網格算符（A 是 2-hop，至少要對角鄰居，即 81 點）、
   且粗網格解要能處理 A 的非正規性。代價高，尚未證明能贏 CGNE。
2. 在那之前，BiCGStab + MG 能用的設定只有「隨機測試向量 + GMRES(4) smoother」（效果全來自 smoother；1e-16 約 111 次，迭代數跳動 45–119）。
3. 值得考慮的替代方向：
   - **Kanamori 2023 結構**（cycle 作用在 D、粗網格 1-hop 的 R D D_PV⁻¹ P、外層仍 BiCGStab 解 A′）。
   - **deflation**：既然粗空間其實就是在抓低模，直接用 Chebyshev 濾出的低模做 deflation（DESIGN §7.7 的方向）。
   - **Chebyshev smoother 取代 GMRES(4)**：固定映射，應該能讓 BiCGStab 的迭代數不再跳動；已實作，在 12⁴ 上還沒量。
4. ~~待修：BiCGStab 在預處理器輸出 NaN 時要報 breakdown。~~ 已修（`40177fd`）。

## Chebyshev smoother 與 Chebyshev 多項式預處理（`mdwf_smoother_compare`，12⁴、M5 = 1.0、A 系統到 1e-16、5 個右手邊）

cycle 是 `mdwf_multigrid` 的參考設定（8 個隨機測試向量、參考粗網格），只換 smoother。
「A 總數」= 外層迭代 ×（2 + 2 × 每次預處理作用 A 的次數）；BiCGStab 每次迭代作用 A 兩次、預處理器兩次。1 次 A = 2 個 Wilson sweep。

| 預處理器 | 每次預處理作用 A | BiCGStab（seed 91–95） | A 總數 | FGMRES(20)（seed 91） |
|---|---|---|---|---|
| cycle，GMRES(4) | 9 | 122 100 104 78 70 | ≈ 1900 | 43（≈ 430） |
| cycle，Chebyshev 5 步、ratio 10 | 9 | 42 38 33 34 35 | ≈ 720 | 59 |
| cycle，Chebyshev 5、ratio 30 | 9 | 31 27 26 29 29 | ≈ 560 | 46 |
| cycle，Chebyshev 9、ratio 30 | 17 | 21 20 20 19 19 | ≈ 720 | 32 |
| cycle，Chebyshev 9、ratio 100 | 17 | 15 14 13 13 14 | ≈ 500 | 24 |
| **只有 Chebyshev 多項式** 9 步、ratio 100 | 8 | 25 24 22 23 26 | ≈ 430 | — |
| 只有多項式 9、ratio 300 | 8 | 26 25 24 24 24 | ≈ 440 | — |
| **只有多項式 17、ratio 300** | 16 | 13 12 12 12 13 | **≈ 420** | — |
| 只有多項式 17、ratio 1000 | 16 | 17 17 16 18 16 | ≈ 570 | — |
| 只有多項式 33、ratio 1000 | 32 | 8 8 8 8 8 | ≈ 530 | — |

（hi = 1.1 |λ_max(A)| = 4.905；物理殘差 1e-14 左右，ratio 1000 的兩列是 3e-16，原因未查。）

- **固定的 Chebyshev smoother 讓 BiCGStab 穩定**：GMRES(4) 的迭代數跳 70–122，Chebyshev 只差 1–2 次。這證實「BiCGStab 不穩是因為 GMRES smoother 讓 M 不固定」。
- **cycle 沒有貢獻**：只用多項式（沒有 setup、沒有粗網格）的 A 總數比 cycle 還少（≈ 420 對 ≈ 500）。cycle 的迭代數較少只是因為前後各平滑一次，等於兩倍次數的多項式。
- **Chebyshev 多項式預處理的 BiCGStab ≈ 420 次 A ≈ 840 sweeps，與 FGMRES + GMRES(4) cycle（≈ 430）相當**，記憶體少很多（BiCGStab 8 個向量 + 多項式 3 個，對 FGMRES(20) 的 40 多個）。
- **仍輸 even-odd CGNE 約 2.2 倍**（CGNE 物理 ~1e-16 約 190 次、380 sweeps）。其中約 2 倍來自 A 作用在全格子。

### 新的方向（建議下一個 session 先做）

1. **even-odd 版的 A′**：D 與 D_PV 都有 even-odd 分解（`Mdwf` 已實作 `EvenOdd`），C 是 site 內的對角區塊。
   把 A′ = X†Y 的兩個因子各自換成 Schur 補（Y_ee = (D C⁻¹) 的 Schur、X 同理），得到半格子上的正定算子，
   預期 sweeps 減半，與 CGNE 打平；配上 Chebyshev 多項式的 BiCGStab 每次迭代只有幾個全域內積（8 次迭代 vs CGNE ~190 次），在多節點/GPU 上是實際優勢。需要先寫設計（正定性要重新確認）。
2. **把 Chebyshev 預處理做成正式選項**：`Chebyshev` 已在函式庫；缺的是自動選 hi（已有 `largest_magnitude`）與 ratio、steps 的預設（目前最好：17 步、ratio 300）。
3. 多重網格（粗網格修正）的問題留著：局部一致性弱、需要精確的 2-hop 粗網格與大量向量，目前看不到能贏多項式預處理的路。

**Chebyshev 已實作並 push（`f72260f`）**：`ChebyshevFilter`、`Chebyshev`、`largest_magnitude`（`src/solver/chebyshev.rs`），
`MdwfSmoother::Chebyshev { steps, ratio }`；測試在 `stage1_solver`（純量算符上逐位等於 Chebyshev 多項式的封閉形式）與
`stage3_mdwf`（BiCGStab + Chebyshev cycle 解 D x = b）。λ_max(A†A) ≈ 20.9（power iteration），|λ_max(A)| ≈ 4.53。

## 尚未跑完（寫這份時）

- （無；`mdwf_lowmodes`、`mdwf_coherence` 都已跑完，結果在上面。）
- 16 個向量的 Petrov-Galerkin 一列沒跑完（已知 Petrov-Galerkin 不可行，停掉了）。
- Chebyshev 與 A′ 的設計草稿在 DESIGN.md §7.6.5，等使用者確認才實作。
