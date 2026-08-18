//! Thin binary shim. Everything lives in the library so it can be tested and
//! reused; see `cli::run`.

fn main() {
    noetic::cli::run();
}
