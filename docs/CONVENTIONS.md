# Conventions

So that numbers can be compared with other codes:

- **Wilson operator.** `D ψ(x) = ψ(x) − κ Σ_μ [ (1 − γ_μ) U_μ(x) ψ(x+μ) + (1 + γ_μ) U_μ(x−μ)† ψ(x−μ) ]`.
  The hopping parameter is related to the bare mass by `κ = 1 / (2 m0 + 8)`.
- **Clover operator.** `D = D_Wilson − κ c_SW Σ_{μ<ν} F_μν γ_μ γ_ν`, with the clover-leaf
  field strength `F_μν = (Q − Q†)/8`, Q being the sum of the four plaquettes in the μ-ν
  plane that start at x.
- **Möbius domain-wall operator.** `D = 1 + Y + W (b − c Y)`, W the Wilson operator with
  mass −M5 and Y the hop in the fifth dimension with the quark mass in its corners, in the
  normalisation of Bridge++'s `Fopr_Domainwall`; b = 1, c = 0 is Shamir's operator. The
  notation is below, with that of the references.
- **Gamma matrices.** Dirac representation (γ4 diagonal), Hermitian, with
  γ5 = [[0, 1], [1, 0]] in 2×2 blocks.
- **Boundary conditions** apply to the fermions only; the gauge field is always periodic.
- **Meson correlator.** `C(t) = Σ_x Tr[ (γ5 Γ_sink) S1(x,t) (Γ_source γ5) S2(x,t)† ]`,
  where S1 and S2 are the propagators of the two quarks from the same source and the trace
  is over spin and colour. No further sign or normalisation is applied.
- **Plaquette.** Normalised so that a unit gauge field gives 1.
- **Coulomb gauge.** Maximises Σ_x Σ_i Re tr U_i(x) over the spatial directions i on every
  time slice. The violation of a time slice is Σ_x |Δ(x)|² / (4 Nc V_t), where Δ(x) is the
  traceless part of M − M† with M = Σ_i [U_i(x−i) − U_i(x)], |·|² sums the squares of all
  real components and V_t is the number of sites in a time slice.

These follow Bridge++, which is what the results are checked against.

## The domain-wall operator and the notation of the references

The code and `docs/mg_background/MDWF_MULTIGRID_NOTE.tex` write, for fields ψ_s(x),
s = 0, …, Ls − 1, and P± = (1 ± γ5)/2:

- `W = (4 − M5) − K/2`, K the hops of the Wilson operator: W is 1/(2κ) times the Wilson
  operator above at κ = 1/(8 − 2 M5) (bare mass −M5). M5 > 0 is the domain-wall height.
- `(Y ψ)_s = −P- ψ_(s+1) − P+ ψ_(s−1)` with `ψ_Ls ≡ −m ψ_0`, `ψ_(−1) ≡ −m ψ_(Ls−1)`: the quark
  mass m is only in the corners.
- `D = 1 + Y + W (b − c Y) = C − (K/2)(b − c Y)`.
- **C**, the part of D within a four-dimensional site: `C = d+ + d- Y`, `d+ = 1 + b (4 − M5)`,
  `d- = 1 − c (4 − M5)`. Lower bidiagonal in s on P+ ψ, upper on P- ψ, with the mass in one
  corner; inverted exactly by a substitution along s (`Mdwf::mult_site_inv`).
- **Q**, Neff's α (`neff_alpha`): `D^(α) = D Q`, `Q_0 = P- + α P+`, `Q_s = α` for 0 < s < Ls − 1,
  `Q_(Ls−1) = P+ + α P-` (`neff_q`; Q⁻¹ is Q at 1/α). The site part of D^(α) is C Q. The
  solution of D ψ = η is ψ = Q ψ^(α) with D^(α) ψ^(α) = η; Q leaves the four-dimensional quark
  `q = P- ψ_0 + P+ ψ_(Ls−1)` unchanged.
- `Γ5 = γ5 R`, (R ψ)_s = ψ_(Ls−1−s).

| | RutticeQCD, the note | Brower, Neff, Orginos (arXiv:1206.5214) | Kanamori, Chen, Matsufuru (LATTICE2024 414) | Bridge++ code (`Fopr_Domainwall`, `_eo`) |
|---|---|---|---|---|
| Domain-wall height | M5 > 0 | M5 = −M5 here, O(−1) (Eq. 2.13) | M0 = M5 | M0 = M5, `domain_wall_height` |
| Fifth dimension | s = 0 … Ls − 1 | s = 1 … Ls | s = 1 … Ns | `is` = 0 … Ns − 1, `extent_of_5th_dimension` |
| Wilson part | W = (4 − M5) − K/2 | D^Wilson(M5) = W (2.13) | D_W(−M0) = W | (1/2κ) times the Wilson kernel, κ = 1/(8 − 2 M0) |
| D± | 1 + b W and 1 − c W | D+ = b D^Wilson + 1, **D- = c D^Wilson − 1** (2.18) | D+ = b D_W + 1, **D- = c D_W − 1** (2) | — |
| D | 1 + Y + W (b − c Y) | D_DW(m) (2.20): the same matrix | D(m) (1): the same matrix | `Fopr_Domainwall`, mode "D" |
| C | d+ + d- Y | — | C = diag(D_ee, D_oo) (4): the same | D_ee, D_oo of `Fopr_Domainwall_eo` (`L_inv`, `U_inv`); `m_dp` = α d+, `m_dm` = α d- |
| Neff's α, Q | `neff_alpha`, D Q | none: its α = (b + c)/(b − c) is the Möbius scale factor | none | `parameter_alpha`: the same Q, in the hops and in C Q; `postProp` returns ψ^(α) |
| Γ5 | γ5 R | γ5 R D-⁻¹ (2.22); H = Γ5 D | — | — |
| Four-dimensional quark | P- ψ_0 + P+ ψ_(Ls−1) | q_L = P- Ψ_1, q_R = P+ Ψ_Ls: the same | — | `mult_chproj_4d` of s = 0 and Ns − 1: the same |

So D, C and the walls are the same everywhere; the differences are names and signs:

- **D-.** In BNO and Kanamori et al. D- = c W − 1, the factor of P± ψ_(s∓1) in D. Here that factor
  is written −(1 − c W) through Y, d- = 1 − c (4 − M5) is minus the site part of their D-, and
  the comments on the source (`Mdwf::source_5d`) call 1 − c W "D₋": that is −D- of the papers.
  The source (1 − c W)(P+ η at s = 0, P- η at Ls − 1) is that of Bridge++'s domain-wall test,
  sign included.
- **M5.** BNO's M5 is the negative Wilson mass (M5 = −1.8, say); here, in Kanamori et al. (M0)
  and in Bridge++ it is the positive height. Kanamori et al. Fig. 1 writes M0 = −1 in the
  caption for the height 1 of the text.
- **α.** BNO's α is the scale factor of the Möbius kernel, (b + c)/(b − c) (2 at b = 1.5, c = 0.5);
  Neff's α (arXiv:1501.04950; Chen, Kanamori, Matsufuru, Neff, ICCSA 2025) is Q above. The
  operator D^(α) here and Bridge++'s with `parameter_alpha` were checked to be the same on 12⁴
  (2026-10-20: the same iterations of every column at α = 0.6 and the same correlators).
- **Γ5.** BNO include D-⁻¹ in Γ5; here Γ5 = γ5 R and the Hermitian operator of the Möbius case
  is Γ5 D-⁻¹ D.
