//! `ruttice-flow`: the Wilson flow of gauge configurations, for the scale t0 and the
//! topological charge of every configuration of a Monte Carlo history.
//!
//! ```text
//! ruttice-flow LX LY LZ LT FORMAT FILE... [--step EPS] [--t-max T] [--skip N]
//! mpirun -np 4 ruttice-flow-mpi LX LY LZ LT FORMAT FILE... [--step EPS] [--t-max T]
//! ```
//!
//! FORMAT is text, binary, nersc, ildg or scidac (`gauge_config_type_input`). For every file
//! one line: the plaquette, t0/a² (t² E = 0.3, E of the clover field strength, interpolated
//! between the steps of the flow), and the topological charge of the clover field strength
//! (Q_clover) and its improved form with rectangles (Q_imp, c_plaq 5/3, c_rect −1/12) at the
//! first step past t0. A configuration whose t² E stays below 0.3 up to the largest flow time
//! has `-` there. At the end, the mean, its error and the integrated autocorrelation time of t0
//! and of the charges over the files in the order given, without the first N (`--skip`,
//! thermalization; `gamma_lines`).

use crate::gauge::flow::{flow_to_t0, AtT0, WilsonFlow, T0_LEVEL};
use crate::gauge::plaquette;
use crate::numerics::gamma_method;
use crate::{Boundary, Comm, Lattice, Stencil, ND};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

fn usage(out: &mut dyn Write) -> u8 {
    let _ = writeln!(
        out,
        "usage: ruttice-flow LX LY LZ LT FORMAT FILE... [--step EPS] [--t-max T] [--skip N]\n\
         FORMAT: text, binary, nersc, ildg or scidac"
    );
    2
}

/// What `ruttice-flow` does with its arguments (the program name left out); the exit code.
pub fn flow_measure(args: &[String], comm: Arc<dyn Comm>, out: &mut dyn Write) -> u8 {
    let (mut step, mut t_max, mut skip) = (0.01, 10.0, 0usize);
    let mut positional = Vec::new();
    let mut k = 0;
    while k < args.len() {
        match args[k].as_str() {
            "--skip" if k + 1 < args.len() => {
                let Ok(n) = args[k + 1].parse::<usize>() else {
                    return usage(out);
                };
                skip = n;
                k += 2;
            }
            "--step" | "--t-max" if k + 1 < args.len() => {
                let Ok(v) = args[k + 1].parse::<f64>() else {
                    return usage(out);
                };
                if args[k] == "--step" {
                    step = v;
                } else {
                    t_max = v;
                }
                k += 2;
            }
            a => {
                positional.push(a.to_string());
                k += 1;
            }
        }
    }
    if positional.len() < 6 || step <= 0.0 {
        return usage(out);
    }
    let mut size = [0usize; ND];
    for (mu, a) in positional[..ND].iter().enumerate() {
        match a.parse() {
            Ok(l) if l > 0 => size[mu] = l,
            _ => return usage(out),
        }
    }
    let Some(format) = super::gauge_format_named(&positional[ND]) else {
        return usage(out);
    };
    let reader = match super::gauge_file(format) {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(out, "{e}");
            return 1;
        }
    };
    let Ok(grid) = super::grid_for(size, None, comm.size()) else {
        let _ = writeln!(
            out,
            "the lattice cannot be split over {} processes",
            comm.size()
        );
        return 1;
    };
    let lat = Lattice::new(size, grid, comm);
    // Every process takes part; rank 0 prints.
    let mut sink = std::io::sink();
    let out: &mut dyn Write = if lat.comm.rank() == 0 { out } else { &mut sink };
    let flow = WilsonFlow { step, exp_order: 8 };
    let _ = writeln!(
        out,
        "# Wilson flow, step {step}, up to t = {t_max}; t0: t^2 E_clover = {T0_LEVEL}\n\
         # file  plaquette  t0/a^2  t  Q_clover(t)  Q_imp(t)"
    );
    let mut code = 0;
    let mut found: Vec<AtT0> = Vec::new();
    for (i, file) in positional[ND + 1..].iter().enumerate() {
        match measure_one(&lat, &*reader, Path::new(file), &flow, t_max) {
            Ok((plaq, at)) => {
                let _ = writeln!(out, "{file}  {plaq:.15}  {}", t0_columns(at.as_ref()));
                if i >= skip {
                    found.extend(at);
                }
            }
            Err(e) => {
                let _ = writeln!(out, "{file}  error: {e}");
                code = 1;
            }
        }
    }
    if skip > 0 {
        let _ = writeln!(out, "# without the first {skip} files:");
    }
    let _ = gamma_lines(out, &found, 1, "files");
    code
}

/// The columns t0/a², t, Q_clover, Q_imp of a measurement, `-` where t0 was not reached.
pub(super) fn t0_columns(at: Option<&AtT0>) -> String {
    match at {
        Some(a) => format!(
            "{:.6}  {:.4}  {:+.6}  {:+.6}",
            a.t0, a.t, a.q_clover, a.q_improved
        ),
        None => "-  -  -  -".into(),
    }
}

/// The Γ-method (Wolff, S_τ = 1.5) for t0, Q_clover and Q_imp of the measurements `found`,
/// made every `spacing` `unit`s: one line each, τ_int in `unit`s.
pub(super) fn gamma_lines(
    out: &mut dyn Write,
    found: &[AtT0],
    spacing: usize,
    unit: &str,
) -> std::io::Result<()> {
    let series: [(&str, Vec<f64>); 3] = [
        ("t0/a^2", found.iter().map(|a| a.t0).collect()),
        ("Q_clover", found.iter().map(|a| a.q_clover).collect()),
        ("Q_imp", found.iter().map(|a| a.q_improved).collect()),
    ];
    for (name, x) in series {
        let Some(g) = gamma_method(&x, 1.5) else {
            continue;
        };
        let s = spacing as f64;
        writeln!(
            out,
            "# {name}: {:.6} +- {:.6}, tau_int {:.2} +- {:.2} {unit} (W = {}{}; {} measurements)",
            g.mean,
            g.error,
            g.tau_int * s,
            g.tau_int_error * s,
            g.window,
            if g.window_found {
                ""
            } else {
                ", window not found: too few"
            },
            x.len()
        )?;
    }
    Ok(())
}

/// The plaquette of one configuration, and what the flow finds at its t0.
fn measure_one(
    lat: &Arc<Lattice>,
    reader: &dyn crate::io::GaugeFile,
    file: &Path,
    flow: &WilsonFlow,
    t_max: f64,
) -> Result<(f64, Option<AtT0>), crate::io::IoError> {
    let mut u = reader.read(file, lat)?;
    let plaq = plaquette(&u, &Stencil::nearest(lat, [Boundary::Periodic; ND]));
    Ok((plaq, flow_to_t0(&mut u, flow, t_max)))
}
