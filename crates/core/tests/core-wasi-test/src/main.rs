//! This test program takes argument(s) that determine which WASI feature to
//! exercise and returns an exit code of 0 for success, 1 for WASI interface
//! failure (which is sometimes expected in a test), and some other code on
//! invalid argument(s).

use std::time::Duration;
mod multiplier {
    wit_bindgen::generate!({
        world: "multiplier",
        path: "wit/multiplier.wit"
    });
}
mod hello {
    wit_bindgen::generate!({
        world: "hello",
        path: "wit/hello.wit"
    });
}

mod ml {
    wit_bindgen::generate!({
        world: "ml",
        path: "wit/ml.wit"
    });
}

mod imagenet_classes;
mod imagenet;

use imagenet::imagenet_openvino_test;



use std::path::Path;

use crate::hello::test::test::gggg2::say_hello;



type Result = std::result::Result<(), Box<dyn std::error::Error>>;


fn main() -> Result {
    let mut args = std::env::args();
    let cmd = args.next().expect("cmd");

    match cmd.as_str() {
        "noop" => (),
        "echo" => {
            eprintln!("echo");
            std::io::copy(&mut std::io::stdin(), &mut std::io::stdout())?;
        }
        "alloc" => {
            let size: usize = args.next().expect("size").parse().expect("size");
            eprintln!("alloc {size}");
            let layout = std::alloc::Layout::from_size_align(size, 8).expect("layout");
            unsafe {
                let p = std::alloc::alloc(layout);
                if p.is_null() {
                    return Err("allocation failed".into());
                }
                // Force allocation to actually happen
                p.read_volatile();
            }
        }
        "read" => {
            let path = args.next().expect("path");
            eprintln!("read {path}");
            std::fs::read(path)?;
        }
        "write" => {
            let path = args.next().expect("path");
            eprintln!("write {path}");
            std::fs::write(path, "content")?;
        }
        "multiply" => {
            let input: i32 = args.next().expect("input").parse().expect("i32");
            eprintln!("multiply {input}");
            let output = multiplier::imports::multiply(input);
            println!("{output}");
        }
        "hello" => {
            let input: String = args.next().expect("input").parse().expect("String");
            let output = say_hello(&input);
            println!("{output}");
        }
        "imagenet" => {
            let path_as_string = args.next().expect("path");
            _ = imagenet_openvino_test(path_as_string);
        }
        "sleep" => {
            let duration =
                Duration::from_millis(args.next().expect("duration_ms").parse().expect("u64"));
            eprintln!("sleep {duration:?}");
            std::thread::sleep(duration);
        }
        "panic" => {
            eprintln!("panic");
            panic!("intentional panic");
        }
        cmd => panic!("unknown cmd {cmd}"),
    };
    Ok(())
}