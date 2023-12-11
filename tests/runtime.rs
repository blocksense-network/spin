#[cfg(feature = "e2e-tests")]
mod runtime_tests {
    use runtime_tests::Config;
    use std::path::PathBuf;

    // TODO: write a proc macro that reads from the tests folder
    // and creates tests for every subdirectory
    macro_rules! test {
        ($ident:ident, $path:literal) => {
            #[test]
            fn $ident() {
                run($path)
            }
        };
    }

    test!(outbound_mysql, "outbound-mysql");
    test!(outbound_mysql_no_permission, "outbound-mysql-no-permission");
    test!(outbound_redis, "outbound-redis");
    test!(outbound_redis_no_permission, "outbound-redis-no-permission");
    test!(sqlite, "sqlite");
    test!(sqlite_no_permission, "sqlite-no-permission");
    test!(key_value, "key-value");
    test!(key_value_no_permission, "key-value-no-permission");

    fn run(name: &str) {
        let spin_binary_path = env!("CARGO_BIN_EXE_spin").into();
        let tests_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/runtime-tests/tests");
        let components_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-components");
        let config = Config {
            spin_binary_path,
            tests_path,
            components_path,
            on_error: runtime_tests::OnTestError::Panic,
        };
        let path = config.tests_path.join(name);
        runtime_tests::bootstrap_and_run(&path, &config)
            .expect("failed to bootstrap runtime tests tests");
    }
}