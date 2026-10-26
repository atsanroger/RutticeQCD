//! Settings read from the environment (`RUTTICE_*`). A variable that is set to something it
//! cannot mean stops the program with a message naming the variable and what it takes, rather
//! than being taken for unset; set to the empty string it is unset.

use std::str::FromStr;

/// The value of `name`, if it is set and not empty.
fn value(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) if v.is_empty() => None,
        Ok(v) => Some(v),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => panic!("{name}: the value is not valid text"),
    }
}

/// `name` read as a `T`, `None` if unset; `takes` says what it takes, for the message.
pub(crate) fn parsed<T: FromStr>(name: &str, takes: &str) -> Option<T> {
    value(name).map(|v| {
        v.trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name}={v}: it takes {takes}"))
    })
}

/// The index in `choices` of the value of `name`, `None` if unset.
pub(crate) fn choice(name: &str, choices: &[&str]) -> Option<usize> {
    value(name).map(|v| {
        choices
            .iter()
            .position(|c| *c == v)
            .unwrap_or_else(|| panic!("{name}={v}: it takes one of {}", choices.join(", ")))
    })
}

/// A switch: `1` on, `0` off, unset off.
pub(crate) fn switch(name: &str) -> bool {
    choice(name, &["0", "1"]) == Some(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Names of their own, so that the tests running at once do not see each other's.
    #[test]
    fn values_are_read_or_refused() {
        std::env::set_var("RUTTICE_TEST_ENV_A", "42");
        assert_eq!(parsed::<usize>("RUTTICE_TEST_ENV_A", "a number"), Some(42));
        std::env::set_var("RUTTICE_TEST_ENV_A", "");
        assert_eq!(parsed::<usize>("RUTTICE_TEST_ENV_A", "a number"), None);
        assert_eq!(parsed::<usize>("RUTTICE_TEST_ENV_UNSET", "a number"), None);
        std::env::set_var("RUTTICE_TEST_ENV_B", "pool");
        assert_eq!(choice("RUTTICE_TEST_ENV_B", &["steal", "pool"]), Some(1));
        std::env::set_var("RUTTICE_TEST_ENV_C", "1");
        assert!(switch("RUTTICE_TEST_ENV_C"));
    }

    #[test]
    #[should_panic(expected = "RUTTICE_TEST_ENV_D=4k: it takes a number")]
    fn a_value_that_does_not_parse_is_refused() {
        std::env::set_var("RUTTICE_TEST_ENV_D", "4k");
        parsed::<usize>("RUTTICE_TEST_ENV_D", "a number");
    }

    #[test]
    #[should_panic(expected = "RUTTICE_TEST_ENV_E=yes: it takes one of 0, 1")]
    fn a_switch_takes_zero_or_one() {
        std::env::set_var("RUTTICE_TEST_ENV_E", "yes");
        switch("RUTTICE_TEST_ENV_E");
    }
}
