# Conventions

So that numbers can be compared with other codes:

- **Wilson operator.** `D ψ(x) = ψ(x) − κ Σ_μ [ (1 − γ_μ) U_μ(x) ψ(x+μ) + (1 + γ_μ) U_μ(x−μ)† ψ(x−μ) ]`.
  The hopping parameter is related to the bare mass by `κ = 1 / (2 m0 + 8)`.
- **Clover operator.** `D = D_Wilson − κ c_SW Σ_{μ<ν} F_μν γ_μ γ_ν`, with the clover-leaf
  field strength `F_μν = (Q − Q†)/8`, Q being the sum of the four plaquettes in the μ-ν
  plane that start at x.
- **Möbius domain-wall operator.** `D = 1 + Y + W (b − c Y)`, W the Wilson operator with
  mass −M5 and Y the hop in the fifth dimension with the quark mass in its corners, in the
  normalisation of Bridge++'s `Fopr_Domainwall`; b = 1, c = 0 is Shamir's operator.
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
