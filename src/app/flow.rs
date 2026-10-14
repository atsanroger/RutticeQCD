//! `ruttice-flow`: the Wilson flow of gauge configurations, for the scale t0 and the
//! topological charge of every configuration of a Monte Carlo history.
//!
//!     ruttice-flow LX LY LZ LT FORMAT FILE... [--step EPS] [--t-max T]
//!
//! FORMAT is text, binary, nersc, ildg or scidac (`gauge_config_type_input`). For every file
//! one line: the plaquette, t0/a² (t² E = 0.3, E of the clover field strength, interpolated
//! between the steps of the flow), and the topological charge of the clover field strength
//! (Q_clover) and its improved form with rectangles (Q_imp, c_plaq 5/3, c_rect −1/12) at the
//! first step past t0. A configuration whose t² E stays below 0.3 up to the largest flow time
//! has `-` there.

use super::config::GaugeFormat;
use crate::gauge::flow::{energy_density_clover, topological_charge, WilsonFlow};
use crate::gauge::plaquette;
use crate::{Boundary, Comm, Lattice, Stencil, ND};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// t² E at the scale t0.
const T0_LEVEL: f64 = 0.3;

fn usage(out: &mut dyn Write) -> u8 {
    let _ = writeln!(
        out,
        "usage: ruttice-flow LX LY LZ LT FORMAT FILE... [--step EPS] [--t-max T]\n\
         FORMAT: text, binary, nersc, ildg or scidac"
    );
    2
}

/// What `ruttice-flow` does with its arguments (the program name left out); the exit code.
pub fn flow_measure(args: &[String], comm: Arc<dyn Comm>, out: &mut dyn Write) -> u8 {
    let (mut step, mut t_max) = (0.01, 10.0);
    let mut positional = Vec::new();
    let mut k = 0;
    while k < args.len() {
        match args[k].as_str() {
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
    let format = match positional[ND].to_ascii_lowercase().as_str() {
        "text" => GaugeFormat::BridgeText,
        "binary" => GaugeFormat::Binary,
        "nersc" => GaugeFormat::Nersc,
        "ildg" => GaugeFormat::Ildg,
        "scidac" => GaugeFormat::Scidac,
        _ => return usage(out),
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
    let flow = WilsonFlow { step, exp_order: 8 };
    let _ = writeln!(
        out,
        "# Wilson flow, step {step}, up to t = {t_max}; t0: t^2 E_clover = {T0_LEVEL}\n\
         # file  plaquette  t0/a^2  t  Q_clover(t)  Q_imp(t)"
    );
    let mut code = 0;
    for file in &positional[ND + 1..] {
        match measure_one(&lat, &*reader, Path::new(file), &flow, t_max) {
            Ok(line) => {
                let _ = writeln!(out, "{file}  {line}");
            }
            Err(e) => {
                let _ = writeln!(out, "{file}  error: {e}");
                code = 1;
            }
        }
    }
    code
}

/// The line of one configuration, after its file name.
fn measure_one(
    lat: &Arc<Lattice>,
    reader: &dyn crate::io::GaugeFile,
    file: &Path,
    flow: &WilsonFlow,
    t_max: f64,
) -> Result<String, crate::io::IoError> {
    let mut u = reader.read(file, lat)?;
    let plaq = plaquette(&u, &Stencil::nearest(lat, [Boundary::Periodic; ND]));
    let (mut t, mut before) = (0.0, 0.0);
    while t < t_max {
        flow.step(&mut u);
        let t_next = t + flow.step;
        let now = t_next * t_next * energy_density_clover(&u, 1.0, 0.0);
        if now >= T0_LEVEL {
            let t0 = t + (T0_LEVEL - before) / (now - before) * flow.step;
            return Ok(format!(
                "{plaq:.15}  {t0:.6}  {t_next:.4}  {:+.6}  {:+.6}",
                topological_charge(&u, 1.0, 0.0),
                topological_charge(&u, 5.0 / 3.0, -1.0 / 12.0)
            ));
        }
        (t, before) = (t_next, now);
    }
    Ok(format!("{plaq:.15}  -  -  -  -"))
}
