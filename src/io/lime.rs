//! Gauge configurations in a LIME container: the ILDG format and the lattice-colour-matrix
//! records of the SciDAC QIO library.
//!
//! A LIME file is a sequence of records. Each has a header of 144 bytes (a magic number, a
//! version, two flags that mark the first and the last record of a message, the length of
//! the data as a 64-bit integer, and the name of the record type in 128 bytes), followed
//! by the data, padded with zeros to a multiple of eight bytes. All integers are
//! big-endian.
//!
//! The records used here:
//!
//! - `ildg-format`: XML with the field type, the precision (32 or 64) and the extents
//! - `ildg-binary-data` or `scidac-binary-data`: the links as big-endian IEEE reals, the
//!   site outermost with x running fastest, then the direction, then the matrix row by row
//! - `scidac-private-file-xml`, `scidac-private-record-xml`: what QIO needs to read the
//!   file back, among it the extents, the precision (F or D) and the size of a site
//! - `scidac-checksum`: two 32-bit sums over the sites; for each site the CRC-32 of its
//!   bytes, rotated left by the site's number modulo 29 for the first sum and modulo 31
//!   for the second, combined with exclusive or
//!
//! Written from the descriptions of the formats (the ILDG file-format specification, the
//! c-lime and QIO documentation). No file written by another program has been read with
//! this yet, and no other program has read a file written here.

use super::{write_on_root, FilePrecision, IoError, LINK_REALS};
use crate::{Field, Gauge, Lattice, ND};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

const MAGIC: u32 = 0x4567_89ab;
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 144;
/// Flags of a record header: the first and the last record of a message.
const MESSAGE_BEGIN: u16 = 1 << 15;
const MESSAGE_END: u16 = 1 << 14;
/// Largest record that is read as text.
const TEXT_LIMIT: u64 = 1 << 20;

const ILDG_FORMAT: &str = "ildg-format";
const ILDG_DATA: &str = "ildg-binary-data";
const SCIDAC_DATA: &str = "scidac-binary-data";
const SCIDAC_FILE: &str = "scidac-private-file-xml";
const SCIDAC_USER_FILE: &str = "scidac-file-xml";
const SCIDAC_RECORD: &str = "scidac-private-record-xml";
const SCIDAC_USER_RECORD: &str = "scidac-record-xml";
const SCIDAC_CHECKSUM: &str = "scidac-checksum";

fn malformed<T>(what: impl Into<String>) -> Result<T, IoError> {
    Err(IoError::Format(what.into()))
}

/// One record of a LIME file: its type and where its data are.
struct Record {
    kind: String,
    offset: u64,
    len: u64,
}

/// The records of a LIME file, in order.
fn records(file: &mut (impl Read + Seek)) -> Result<Vec<Record>, IoError> {
    let end = file.seek(SeekFrom::End(0))?;
    let header = HEADER_BYTES as u64;
    let (mut at, mut found) = (0u64, Vec::new());
    while at + header <= end {
        file.seek(SeekFrom::Start(at))?;
        let mut h = [0u8; HEADER_BYTES];
        file.read_exact(&mut h)?;
        if u32::from_be_bytes([h[0], h[1], h[2], h[3]]) != MAGIC {
            return malformed(format!("no LIME record header at byte {at}"));
        }
        let mut len = [0u8; 8];
        len.copy_from_slice(&h[8..16]);
        let len = u64::from_be_bytes(len);
        let name = &h[16..];
        let name = &name[..name.iter().position(|b| *b == 0).unwrap_or(name.len())];
        let offset = at + header;
        if len > end - offset {
            return malformed(format!("the LIME record at byte {at} runs past the end"));
        }
        found.push(Record {
            kind: String::from_utf8_lossy(name).into_owned(),
            offset,
            len,
        });
        at = offset + len.div_ceil(8) * 8;
    }
    if found.is_empty() {
        return malformed("not a LIME file");
    }
    Ok(found)
}

/// The data of a record as text.
fn text(file: &mut (impl Read + Seek), record: &Record) -> Result<String, IoError> {
    if record.len > TEXT_LIMIT {
        return malformed(format!("the record {} is too long for text", record.kind));
    }
    let len = usize::try_from(record.len).expect("below the limit for text");
    let mut data = vec![0u8; len];
    file.seek(SeekFrom::Start(record.offset))?;
    file.read_exact(&mut data)?;
    Ok(String::from_utf8_lossy(&data).into_owned())
}

/// The content of the first element `<name>` of an XML text. Enough for the flat records
/// of these formats; an element with attributes is found too.
fn element<'a>(xml: &'a str, name: &str) -> Option<&'a str> {
    let open = xml
        .match_indices(&format!("<{name}"))
        .find(|(i, m)| matches!(xml[i + m.len()..].chars().next(), Some('>' | ' ')))?
        .0;
    let start = open + xml[open..].find('>')? + 1;
    let len = xml[start..].find(&format!("</{name}>"))?;
    Some(xml[start..start + len].trim())
}

/// A required element of a record.
fn required<'a>(xml: &'a str, name: &str, record: &str) -> Result<&'a str, IoError> {
    element(xml, name).ok_or_else(|| IoError::Format(format!("{record} has no <{name}>")))
}

fn number(text: &str, name: &str) -> Result<usize, IoError> {
    text.parse()
        .map_err(|_| IoError::Format(format!("<{name}> is not a number: {text:?}")))
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// CRC-32 as in zlib, which is what QIO uses.
fn crc32(bytes: &[u8]) -> u32 {
    let mut c = !0u32;
    for b in bytes {
        c = CRC_TABLE[((c ^ u32::from(*b)) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// The SciDAC checksum of a field: see the top of the file.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Checksum {
    a: u32,
    b: u32,
}

impl Checksum {
    /// Add the site with number `rank` in the global lattice, whose bytes in the file are
    /// `bytes`.
    fn add(&mut self, rank: u64, bytes: &[u8]) {
        let crc = crc32(bytes);
        let turns = |modulus: u64| u32::try_from(rank % modulus).expect("below 32");
        self.a ^= crc.rotate_left(turns(29));
        self.b ^= crc.rotate_left(turns(31));
    }

    /// The sums over the sites of all processes from the sums over the sites of each:
    /// exclusive or is the parity of a count, and counts can be added across processes.
    fn across_processes(self, lat: &Lattice) -> Self {
        let mut bits = [0.0f64; 64];
        for i in 0..32 {
            bits[i] = f64::from((self.a >> i) & 1);
            bits[32 + i] = f64::from((self.b >> i) & 1);
        }
        lat.comm.sum(&mut bits);
        let word = |bits: &[f64]| {
            bits.iter().enumerate().fold(0u32, |w, (i, count)| {
                w | (u32::from(count % 2.0 != 0.0) << i)
            })
        };
        Self {
            a: word(&bits[..32]),
            b: word(&bits[32..]),
        }
    }
}

/// What the records before the binary data say about it.
struct Layout {
    dims: [usize; ND],
    precision: FilePrecision,
}

fn layout(file: &mut (impl Read + Seek), records: &[Record]) -> Result<Layout, IoError> {
    let find = |kind: &str| records.iter().find(|r| r.kind == kind);
    if let Some(record) = find(ILDG_FORMAT) {
        let xml = text(file, record)?;
        let field = required(&xml, "field", ILDG_FORMAT)?;
        if field != "su3gauge" {
            return malformed(format!("the ILDG field is {field:?}, not su3gauge"));
        }
        let precision = match required(&xml, "precision", ILDG_FORMAT)? {
            "32" => FilePrecision::Single,
            "64" => FilePrecision::Double,
            other => return malformed(format!("ILDG precision {other:?} is not 32 or 64")),
        };
        let mut dims = [0; ND];
        for (d, name) in dims.iter_mut().zip(["lx", "ly", "lz", "lt"]) {
            *d = number(required(&xml, name, ILDG_FORMAT)?, name)?;
        }
        return Ok(Layout { dims, precision });
    }
    let (Some(of_file), Some(of_record)) = (find(SCIDAC_FILE), find(SCIDAC_RECORD)) else {
        return malformed(format!(
            "the file has neither {ILDG_FORMAT} nor {SCIDAC_FILE} with {SCIDAC_RECORD}"
        ));
    };
    let xml = text(file, of_file)?;
    let listed: Vec<&str> = required(&xml, "dims", SCIDAC_FILE)?
        .split_whitespace()
        .collect();
    if listed.len() != ND {
        return malformed(format!("{SCIDAC_FILE} lists {} extents", listed.len()));
    }
    let mut dims = [0; ND];
    for (d, text) in dims.iter_mut().zip(listed) {
        *d = number(text, "dims")?;
    }
    let xml = text(file, of_record)?;
    let precision = match required(&xml, "precision", SCIDAC_RECORD)? {
        "F" => FilePrecision::Single,
        "D" => FilePrecision::Double,
        other => return malformed(format!("SciDAC precision {other:?} is not F or D")),
    };
    // A gauge field is four colour matrices per site.
    let colors = number(required(&xml, "colors", SCIDAC_RECORD)?, "colors")?;
    let count = number(required(&xml, "datacount", SCIDAC_RECORD)?, "datacount")?;
    let size = number(required(&xml, "typesize", SCIDAC_RECORD)?, "typesize")?;
    if colors != 3 || count != ND || size != LINK_REALS * precision.bytes() {
        return malformed(format!(
            "the SciDAC record holds {count} objects of {size} bytes with {colors} colours \
             per site, not the four links of a gauge field"
        ));
    }
    Ok(Layout { dims, precision })
}

/// Read a gauge configuration from a LIME file: ILDG, or a lattice-colour-matrix record
/// written by QIO. The checksum is verified if the file has one.
///
/// With several processes, each reads the lines of the lattice it holds a part of.
pub fn read_lime(path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError> {
    let mut file = BufReader::new(File::open(path)?);
    let records = records(&mut file)?;
    let Layout { dims, precision } = layout(&mut file, &records)?;
    if dims != lat.global {
        return Err(IoError::Lattice {
            file: dims,
            expected: lat.global,
        });
    }
    let Some(data) = records
        .iter()
        .find(|r| r.kind == ILDG_DATA || r.kind == SCIDAC_DATA)
    else {
        return malformed(format!(
            "the file has neither {ILDG_DATA} nor {SCIDAC_DATA}"
        ));
    };
    let real = precision.bytes();
    let site_bytes = ND * LINK_REALS * real;
    if data.len != (lat.num_global_sites() * site_bytes) as u64 {
        return malformed(format!(
            "the binary record has {} bytes, the lattice needs {}",
            data.len,
            lat.num_global_sites() * site_bytes
        ));
    }

    let mut u = Field::<f64, Gauge>::gauge(lat);
    let num_sites = lat.num_sites();
    let out = u.as_mut_slice();
    let [lx, ly, lz, lt] = lat.global;
    let mut line = vec![0u8; lx * site_bytes];
    let mut sum = Checksum::default();
    for t in 0..lt {
        for z in 0..lz {
            for y in 0..ly {
                // A line along x is on this process in part or not at all.
                if lat.local_site([lat.origin[0], y, z, t]).is_none() {
                    continue;
                }
                let first = lx * (y + ly * (z + lz * t));
                file.seek(SeekFrom::Start(data.offset + (first * site_bytes) as u64))?;
                file.read_exact(&mut line)?;
                for (x, bytes) in line.chunks_exact(site_bytes).enumerate() {
                    let Some(site) = lat.local_site([x, y, z, t]) else {
                        continue;
                    };
                    sum.add((first + x) as u64, bytes);
                    for (k, value) in bytes.chunks_exact(real).enumerate() {
                        let (mu, c) = (k / LINK_REALS, k % LINK_REALS);
                        out[c + LINK_REALS * (site + num_sites * mu)] = match precision {
                            FilePrecision::Single => {
                                f64::from(f32::from_be_bytes(value.try_into().expect("four bytes")))
                            }
                            FilePrecision::Double => {
                                f64::from_be_bytes(value.try_into().expect("eight bytes"))
                            }
                        };
                    }
                }
            }
        }
    }

    // Every process makes the global sum, whether or not it finds a checksum to compare.
    let computed = sum.across_processes(lat);
    if let Some(record) = records.iter().find(|r| r.kind == SCIDAC_CHECKSUM) {
        let xml = text(&mut file, record)?;
        let hex = |name: &str| -> Result<u32, IoError> {
            let text = required(&xml, name, SCIDAC_CHECKSUM)?;
            u32::from_str_radix(text, 16)
                .map_err(|_| IoError::Format(format!("<{name}> is not hexadecimal: {text:?}")))
        };
        let recorded = Checksum {
            a: hex("suma")?,
            b: hex("sumb")?,
        };
        if recorded != computed {
            return Err(IoError::Checksum {
                recorded: (recorded.a, recorded.b),
                computed: (computed.a, computed.b),
            });
        }
    }
    Ok(u)
}

/// Write the header of a record with `len` bytes of data.
fn header(out: &mut impl Write, kind: &str, len: u64, flags: u16) -> Result<(), IoError> {
    let mut h = [0u8; HEADER_BYTES];
    h[0..4].copy_from_slice(&MAGIC.to_be_bytes());
    h[4..6].copy_from_slice(&VERSION.to_be_bytes());
    h[6..8].copy_from_slice(&flags.to_be_bytes());
    h[8..16].copy_from_slice(&len.to_be_bytes());
    // The names used here are far shorter than the 127 bytes there is room for.
    h[16..16 + kind.len()].copy_from_slice(kind.as_bytes());
    out.write_all(&h)?;
    Ok(())
}

/// Zeros up to the next multiple of eight bytes after `len` bytes of data.
fn pad(out: &mut impl Write, len: u64) -> Result<(), IoError> {
    let rest = usize::try_from(len.div_ceil(8) * 8 - len).expect("below eight");
    out.write_all(&[0u8; 8][..rest])?;
    Ok(())
}

/// Write a whole record of text.
fn text_record(out: &mut impl Write, kind: &str, xml: &str, flags: u16) -> Result<(), IoError> {
    let len = xml.len() as u64;
    header(out, kind, len, flags)?;
    out.write_all(xml.as_bytes())?;
    pad(out, len)
}

/// The date as QIO writes it, e.g. `Wed Oct  7 06:30:00 2026 UTC`.
fn date() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Days since 1970-01-01 to a date of the Gregorian calendar, counting from 1 March
    // of year 0 in eras of 400 years (146097 days).
    let shifted = days + 719_468;
    let (era, day_of_era) = (shifted / 146_097, shifted % 146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = (month_from_march + 2) % 12;
    let year = year_of_era + era * 400 + u64::from(month < 2);
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    // 1970-01-01 was a Thursday.
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    format!(
        "{} {} {day:2} {:02}:{:02}:{:02} {year} UTC",
        DAYS[usize::try_from(days % 7).expect("below seven")],
        MONTHS[usize::try_from(month).expect("below twelve")],
        rest / 3_600,
        rest / 60 % 60,
        rest % 60
    )
}

/// Which records carry the binary data and describe it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flavour {
    Ildg,
    Scidac,
}

const XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>"#;

fn write(
    path: &Path,
    u: &Field<f64, Gauge>,
    precision: FilePrecision,
    flavour: Flavour,
) -> Result<(), IoError> {
    let lat = u.lattice();
    let [lx, ly, lz, _] = lat.global;
    let real = precision.bytes();
    let site_bytes = ND * LINK_REALS * real;
    let len = (lat.num_global_sites() * site_bytes) as u64;
    // Rank 0 writes the file and computes the checksum as the slices come by.
    write_on_root(
        u,
        || {
            let mut out = BufWriter::new(File::create(path)?);
            begin(&mut out, lat.global, precision, flavour, len)?;
            Ok((out, Checksum::default()))
        },
        |(out, sum), t, slice| {
            let first = (t * lx * ly * lz) as u64;
            let mut bytes = vec![0u8; site_bytes];
            for (k, site) in slice
                .as_chunks::<{ ND * LINK_REALS }>()
                .0
                .iter()
                .enumerate()
            {
                for (value, to) in site.iter().zip(bytes.chunks_exact_mut(real)) {
                    match precision {
                        // Rounding to single precision is what writing in it means.
                        FilePrecision::Single => to.copy_from_slice(&(*value as f32).to_be_bytes()),
                        FilePrecision::Double => to.copy_from_slice(&value.to_be_bytes()),
                    }
                }
                sum.add(first + k as u64, &bytes);
                out.write_all(&bytes)?;
            }
            Ok(())
        },
        |(mut out, sum)| {
            pad(&mut out, len)?;
            let checksum = format!(
                "{XML}<scidacChecksum><version>1.0</version><suma>{:x}</suma>\
                 <sumb>{:x}</sumb></scidacChecksum>",
                sum.a, sum.b
            );
            text_record(&mut out, SCIDAC_CHECKSUM, &checksum, MESSAGE_END)?;
            out.flush()?;
            Ok(())
        },
    )
}

/// The records before the links, and the header of the record of the links (`len`
/// bytes).
fn begin(
    out: &mut impl Write,
    dims: [usize; ND],
    precision: FilePrecision,
    flavour: Flavour,
    len: u64,
) -> Result<(), IoError> {
    let [lx, ly, lz, lt] = dims;
    let real = precision.bytes();

    // The first message describes the file, the second holds the field.
    let of_file = format!(
        "{XML}<scidacFile><version>1.1</version><spacetime>{ND}</spacetime>\
         <dims>{lx} {ly} {lz} {lt} </dims><volfmt>0</volfmt></scidacFile>"
    );
    text_record(out, SCIDAC_FILE, &of_file, MESSAGE_BEGIN)?;
    let user_file = format!("{XML}<info>gauge configuration written by RutticeQCD</info>");
    text_record(out, SCIDAC_USER_FILE, &user_file, MESSAGE_END)?;

    let (letter, bits) = match precision {
        FilePrecision::Single => ('F', 32),
        FilePrecision::Double => ('D', 64),
    };
    let of_record = format!(
        "{XML}<scidacRecord><version>1.1</version><date>{}</date><recordtype>0</recordtype>\
         <datatype>QDP_{letter}3_ColorMatrix</datatype><precision>{letter}</precision>\
         <colors>3</colors><typesize>{}</typesize><datacount>{ND}</datacount></scidacRecord>",
        date(),
        LINK_REALS * real
    );
    text_record(out, SCIDAC_RECORD, &of_record, MESSAGE_BEGIN)?;
    let user_record = format!("{XML}<info>SU(3) gauge field, four links per site</info>");
    text_record(out, SCIDAC_USER_RECORD, &user_record, 0)?;
    let data_kind = match flavour {
        Flavour::Scidac => SCIDAC_DATA,
        Flavour::Ildg => {
            let format = format!(
                "{XML}<ildgFormat xmlns=\"http://www.lqcd.org/ildg\" \
                 xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" \
                 xsi:schemaLocation=\"http://www.lqcd.org/ildg \
                 http://www.lqcd.org/ildg/filefmt.xsd\"><version>1.0</version>\
                 <field>su3gauge</field><precision>{bits}</precision>\
                 <lx>{lx}</lx><ly>{ly}</ly><lz>{lz}</lz><lt>{lt}</lt></ildgFormat>"
            );
            text_record(out, ILDG_FORMAT, &format, 0)?;
            ILDG_DATA
        }
    };

    header(out, data_kind, len, 0)
}

/// Write a gauge configuration as an ILDG file, the way QIO writes one: the SciDAC records
/// that describe the file and the record, `ildg-format`, `ildg-binary-data` and the
/// checksum.
pub fn write_ildg(
    path: &Path,
    u: &Field<f64, Gauge>,
    precision: FilePrecision,
) -> Result<(), IoError> {
    write(path, u, precision, Flavour::Ildg)
}

/// Write a gauge configuration as a lattice-colour-matrix record of QIO, with the data in
/// `scidac-binary-data`.
pub fn write_scidac(
    path: &Path,
    u: &Field<f64, Gauge>,
    precision: FilePrecision,
) -> Result<(), IoError> {
    write(path, u, precision, Flavour::Scidac)
}
