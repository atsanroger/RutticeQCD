//! The keywords of the parameter files.

use super::*;

keyword_enum! {
    /// Where the fields live and how the kernels run.
    pub enum Backend {
        Cpu => ["cpu"],
        /// Several sites in the lanes of one SIMD register (`simd_lanes`).
        Simd => ["simd"],
        /// The CUDA kernels extracted from Bridge++ (`backend::cuda_bridge`; a program built
        /// with the feature `cuda-bridge`).
        CudaBridge => ["cuda_bridge"],
        /// Not implemented yet: the keyword is reserved.
        Gpu => ["gpu", "cuda"],
    }
}

keyword_enum! {
    pub enum VerboseLevel {
        Crucial => ["Crucial"],
        General => ["General"],
        Detailed => ["Detailed"],
        Paranoiac => ["Paranoiac"],
    }
}

keyword_enum! {
    pub enum GaugeType {
        /// One real per line, the Bridge++ text format.
        Text => ["Text", "Bridge_text", "Text_4x4x4x8"],
        /// Big-endian doubles without a header, the Bridge++ binary format.
        Binary => ["Binary"],
        /// The NERSC format: a text header, then the links.
        Nersc => ["NERSC"],
        Ildg => ["ILDG"],
        /// A lattice-colour-matrix record of the SciDAC QIO library, in a LIME container.
        Scidac => ["SciDAC", "QIO"],
        /// The dataset `gauge` of an HDF5 file (a program built with the feature `hdf5`).
        Hdf5 => ["HDF5"],
    }
}

impl From<GaugeType> for GaugeFormat {
    fn from(t: GaugeType) -> Self {
        match t {
            GaugeType::Text => GaugeFormat::BridgeText,
            GaugeType::Binary => GaugeFormat::Binary,
            GaugeType::Nersc => GaugeFormat::Nersc,
            GaugeType::Ildg => GaugeFormat::Ildg,
            GaugeType::Scidac => GaugeFormat::Scidac,
            GaugeType::Hdf5 => GaugeFormat::Hdf5,
        }
    }
}

keyword_enum! {
    pub enum FermionType {
        /// Möbius domain-wall fermions with a Wilson kernel (b = 1, c = 0: Shamir).
        DomainWall => [
            "Domainwall_5din",
            "Domainwall",
            "Domain_wall",
            "DWF",
            "MDWF",
            "Mobius",
            "Mobius_DWF",
            "Mobius_domainwall",
            "Mobius_domain_wall",
            "Domainwall_Mobius",
        ],
        /// Wilson fermions (`hopping_parameter`).
        Wilson => ["Wilson"],
        /// Wilson fermions with the clover term (`hopping_parameter`, `clover_coefficient`).
        Clover => ["Clover", "Wilson_clover", "Clover_Wilson"],
    }
}

impl FermionType {
    pub fn is_domain_wall(self) -> bool {
        self == FermionType::DomainWall
    }
}

keyword_enum! {
    pub enum GammaType {
        Dirac => ["Dirac"],
        Chiral => ["Chiral"],
    }
}

keyword_enum! {
    /// Which operator the solver works on, on the even sites.
    pub enum OperatorType {
        /// A′ = (D_PV C_PV⁻¹)† D C⁻¹ in its even-odd form; Neff's α cancels in it.
        PvRight => ["PV_right", "PV", "A_prime", "Aprime", "right"],
        /// The left-preconditioned even-odd A′, which sees α.
        PvLeft => ["PV_left", "left"],
        /// The normal equations L† L of the left Schur complement: CGNE, which sees α.
        NormalLeft => ["Normal_left", "Normal", "CGNE"],
        /// Wilson and clover fermions: the Schur complement on the even sites (CG: its
        /// normal equations). With the multigrid, the coarse system is solved on the even
        /// coarse sites too.
        EvenOdd => ["Even_odd", "eo", "Standard_eo", "Schur"],
        /// Wilson and clover fermions: D on the whole lattice (CG: its normal equations),
        /// and with the multigrid the coarse system on the whole coarse lattice.
        Lexical => ["Lexical", "Full", "Standard", "Whole"],
    }
}

keyword_enum! {
    pub enum SolverType {
        /// Flexible GCR keeping `number_of_kept_directions` directions.
        Gcr => ["GCR"],
        /// Block GCR: the right-hand sides of a solve share the kept directions.
        BlockGcr => ["Block_GCR", "BlockGCR"],
        BiCgStab => ["BiCGStab", "BiCGStab_Cmplx", "BiCG_Stab"],
        /// Flexible GMRES; `number_of_kept_directions` is the size of its Krylov space.
        Fgmres => ["FGMRES", "GMRES"],
        /// Conjugate gradients, on the normal equations (`operator_type: Normal_left`).
        Cg => ["CG", "CGNE", "CG_normal"],
        /// MINRES on a Γ5-Hermitian form: for the second coarse factor only.
        Minres => ["MINRES"],
    }
}

keyword_enum! {
    pub enum PreconditionerType {
        None => ["None", "No"],
        /// A Chebyshev polynomial of the operator (block `Chebyshev`).
        Chebyshev => ["Chebyshev"],
        /// The even-odd Pauli-Villars multigrid (block `MultiGrid`).
        Multigrid => ["MG", "MultiGrid", "Multi_Grid"],
    }
}

keyword_enum! {
    pub enum PrecisionType {
        Single => ["single", "float", "f32"],
        Double => ["double", "f64"],
    }
}

keyword_enum! {
    pub enum SourceType {
        /// A unit source at one site, twelve columns (colour × spin).
        Local => ["Local", "Point"],
        /// Wilson and clover: smeared sources on the time slice of `source_position`, the
        /// wave function exp(−slope r^power), normalised (`slope`, `power`). Gauge dependent:
        /// see `GaugeFixing`.
        Exponential => ["Exponential"],
    }
}

keyword_enum! {
    pub enum GaugeFixingType {
        Coulomb => ["Coulomb"],
    }
}

keyword_enum! {
    pub enum SmootherType {
        /// A Chebyshev polynomial of Ã (domain-wall).
        Chebyshev => ["Chebyshev"],
        /// The Schwarz alternating procedure on the blocks (Wilson, clover).
        Sap => ["SAP", "Schwarz"],
        /// GMRES(m) on D (Wilson, clover).
        Gmres => ["GMRES"],
    }
}
