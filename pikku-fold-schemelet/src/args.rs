use crate::config::DEFAULT_LOG_M;
use std::env;
use std::process;

pub(crate) struct Args {
    pub(crate) log_m: usize,
    pub(crate) skip_output_check: bool,
}

pub(crate) fn parse_args() -> Args {
    let mut log_m = DEFAULT_LOG_M;
    let mut skip_output_check = false;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--log-m" => {
                let value = it.next().unwrap_or_else(|| {
                    eprintln!("--log-m needs a value");
                    process::exit(2);
                });
                log_m = value.parse().unwrap_or_else(|_| {
                    eprintln!("--log-m must be an integer");
                    process::exit(2);
                });
            }
            "--skip-output-check" => skip_output_check = true,
            _ => {
                eprintln!("unknown argument: {arg}");
                process::exit(2);
            }
        }
    }
    Args {
        log_m,
        skip_output_check,
    }
}
