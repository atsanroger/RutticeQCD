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
| 3 | Chebyshev 測試向量（§7.6.5 (b)）：對 H = A′†A′ 低通濾波，`ChebyshevFilter` 放 `src/solver/` | 待使用者確認 (b) | 半天（與 4 合計） |
| 4 | Chebyshev smoother（§7.6.5 (c)）：非厄米 Chebyshev 迭代，`MdwfSmoother::Chebyshev` | 待使用者確認 (c) | 同上 |
| 5 | 粗網格算符（§7.6.5 (d)）：步驟 1 說不是截斷 → 「保留 81 點」不會有用；要查的是投影本身（見下的推測），或考慮 Kanamori 2023 結構（cycle 作用在 D） | 待決定 | 不確定（研究問題） |
| 6 | 驗收：12⁴、16⁴，M5 = 1.0，5 個右手邊，物理殘差 ≈ 1e-16，BiCGStab + MG 的細網格 sweep 總數 < even-odd CGNE（12⁴ 約 400） | — | 1–2 小時 |

成功標準：BiCGStab（不是 FGMRES）+ 多重網格，粗網格修正真的有作用（FGMRES 的外層數明顯低於「只做平滑」的 43、平滑誤差的修正接近 √(1-|PP†s|²)），且細網格 sweep 總數贏 CGNE。

## 步驟 1 的結果：精確 P†AP（12⁴，舊 A；A 與 A′ 在粗網格上表現相同）

| 8 個向量，反迭代 1e-6，Galerkin | e = P c 修正後剩 | e = s 修正後剩 | 理論下限 |
|---|---|---|---|
| 探測（截到 8 個最近鄰） | 0.033 | 0.995 | 0.926 |
| **精確 P†AP（不截斷）** | **0.000** | **0.995** | 0.926 |

不截斷時 span(P) 裡的誤差完全修掉，平滑誤差卻一樣沒動 → **不是截斷**。
推測（未驗證）：s 落在 span(P) 的 38% 是逐區塊投影的 PP†s，在區塊邊界不連續、帶大量高模，A(PP†s) 不小，
Galerkin 修正因此不指向 s。對厄米算符這靠 smoother 或 smoothed aggregation（先平滑 P）補救；
但 cycle 加平滑也沒有改善（平滑誤差都剩約 0.707）。下一步可以試：
(1) 量 |A PP†s| 對 |A s|，確認上面的推測；(2) smoothed aggregation：P ← (1 − ωA) P 或 S P；
(3) Kanamori 2023 的結構。Petrov-Galerkin（Q 來自 A† 反迭代）讓粗網格算符本身壞掉（e = P c 剩 0.996），不是解法。

步驟 2 附帶的兩個數字（Bridge++ 4³×8 組態，`stage3_mdwf`）：BiCGStab 解 A′ 496 次（舊 A 443 次，**慢 12%**，
與 12⁴ 的 −4% 方向相反 → 速度不是改用 A′ 的理由）；BiCGStab + cycle（Ls 4、m 0.1）62–71 次（舊 A 356–488 次），
原因未確認，可能只是 BiCGStab 配變動預處理器的跳動。

## 尚未跑完（寫這份時）

- `mdwf_coarse_check` 的 16 個向量兩列（輸出在 as13 的 `/tmp/claude-25116/-home-chen-test/f082ac9a-b2e3-41dd-8bad-d9b0129a9044/tasks/b5mrfmw39.output`，暫存檔，可能會被清掉；要重跑：`./target/release/examples/mdwf_coarse_check /home/chen/test/RUQCD/bench/hmc12/conf_quench_12x12x12x12.txt 12 1.0 0.005 8 3`，約 1 小時）。
- Chebyshev 與 A′ 的設計草稿在 DESIGN.md §7.6.5，等使用者確認才實作。
