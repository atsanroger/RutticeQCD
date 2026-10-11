//! Keywords of the parameter files as Bridge++ reads them: case and the separators `_`,
//! `-` and spaces do not matter, so `BiCGStab`, `bicgstab` and `BICG_STAB` are the same,
//! and several spellings may name one choice (`CGNE` and `CG_normal`).

/// The canonical form of a keyword: lower case, without `_`, `-` and spaces.
pub fn normalise(word: &str) -> String {
    word.chars()
        .filter(|c| !matches!(c, '_' | '-' | ' '))
        .flat_map(char::to_lowercase)
        .collect()
}

/// An enum read from a keyword. `keyword_enum!` writes the type, its `Deserialize` and
/// `Display` (the first spelling of each variant), and the list of spellings for messages.
#[macro_export]
#[doc(hidden)]
macro_rules! keyword_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$vmeta:meta])*
                $variant:ident => [$($spelling:literal),+ $(,)?]
            ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        $vis enum $name {
            $(
                $(#[$vmeta])*
                $variant,
            )+
        }

        impl $name {
            /// The spellings accepted, the first of each choice being how it is printed.
            pub const SPELLINGS: &'static [(&'static str, $name)] = &[
                $($(($spelling, $name::$variant),)+)+
            ];

            /// The choice a keyword names, whatever its case and separators.
            pub fn from_keyword(word: &str) -> Option<Self> {
                let key = $crate::app::keyword::normalise(word);
                Self::SPELLINGS
                    .iter()
                    .find(|(s, _)| $crate::app::keyword::normalise(s) == key)
                    .map(|(_, v)| *v)
            }

            /// The canonical spelling.
            pub fn keyword(self) -> &'static str {
                Self::SPELLINGS
                    .iter()
                    .find(|(_, v)| *v == self)
                    .map(|(s, _)| *s)
                    .expect("every choice has a spelling")
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.keyword())
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let word = String::deserialize(d)?;
                Self::from_keyword(&word).ok_or_else(|| {
                    let mut choices: Vec<&str> = Vec::new();
                    for (s, v) in Self::SPELLINGS {
                        if Self::SPELLINGS.iter().find(|(_, w)| w == v).map(|(t, _)| t) == Some(s) {
                            choices.push(s);
                        }
                    }
                    serde::de::Error::custom(format!(
                        "unknown {} {word:?}; one of {} (any case, with or without _)",
                        stringify!($name),
                        choices.join(", ")
                    ))
                })
            }
        }
    };
}

#[cfg(test)]
mod tests {
    keyword_enum! {
        enum Sample {
            Cgne => ["CGNE", "CG_normal", "CG"],
            BiCgStab => ["BiCGStab", "BiCGStab_Cmplx"],
        }
    }

    #[test]
    fn case_and_separators_do_not_matter() {
        for word in ["CGNE", "cgne", "Cg_Ne", "cg-normal", "CG"] {
            assert_eq!(Sample::from_keyword(word), Some(Sample::Cgne), "{word}");
        }
        assert_eq!(
            Sample::from_keyword("bicgstab_cmplx"),
            Some(Sample::BiCgStab)
        );
        assert_eq!(Sample::from_keyword("gmres"), None);
        assert_eq!(Sample::BiCgStab.to_string(), "BiCGStab");
        let parsed: Sample = serde_yaml_ng::from_str("bicg_stab").unwrap();
        assert_eq!(parsed, Sample::BiCgStab);
        let message = serde_yaml_ng::from_str::<Sample>("gmres")
            .unwrap_err()
            .to_string();
        assert!(message.contains("CGNE, BiCGStab"), "{message}");
    }
}
