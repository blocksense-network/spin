## HOW TO RUN IMAGENET HTTP demo

#In main spin repo build spin in `release`

`spin$ cargo build --release`

Go in http examples directory
`cd spin/examples/http-rust-imagenet`
Make `.spin` dir, if it does not exists:
`mkdir .spin`

Trigger the example. First time it will report an error, because a subdirrectory `imagebet` in `.spin` does not existsL
`spin/examples/http-rust-imagenet$ ../../target/release/spin up --build`

```
Building component hello with `cargo build --target wasm32-wasi --release`
    Finished `release` profile [optimized] target(s) in 0.06s
warning: the following packages contain code that will be rejected by a future version of Rust: buf_redux v0.8.4, multipart v0.18.0, traitobject v0.1.0
note: to see what the problems were, use the option `--future-incompat-report`, or run `cargo report future-incompatibilities --id 2`
Finished building all Spin components
Logging component stdio to ".spin/logs/"
Storing default key-value data to ".spin/sqlite_key_value.db"

Serving http://127.0.0.1:3000
Available Routes:
  hello: http://127.0.0.1:3000 (wildcard)
    A simple component that returns hello.
2024-08-13T14:31:42.521207Z ERROR spin_trigger_http: Error processing request: TriggerLoader only supports directory mounts; "/tmp/spinup-ZfODZu/assets/hello" is not a directory    
```

Run it again and you will get:

`spin/examples/http-rust-imagenet$ ../../target/release/spin up --build`

```
Building component hello with `cargo build --target wasm32-wasi --release`
    Finished `release` profile [optimized] target(s) in 0.06s
warning: the following packages contain code that will be rejected by a future version of Rust: buf_redux v0.8.4, multipart v0.18.0, traitobject v0.1.0
note: to see what the problems were, use the option `--future-incompat-report`, or run `cargo report future-incompatibilities --id 2`
Finished building all Spin components
Logging component stdio to ".spin/logs/"
Storing default key-value data to ".spin/sqlite_key_value.db"

Serving http://127.0.0.1:3000
Available Routes:
  hello: http://127.0.0.1:3000 (wildcard)
    A simple component that returns hello.
```

Go to to your browser `http://127.0.0.1:3000/imagenet` and upload an image using the form.

Sample output in the console, with an image of a whiskey bottle
```
Created tensor with ID: Tensor { handle: Resource { handle: 1 } } `Tensor::new` took 68.74 µs
Obtaining output `GraphExecutionContext::get_output` took 29.27 µs
Copying data from tensor. `Tensor::data+dimensions+type` took 23.46 µs
0.12 -> power drill
0.05 -> water bottle
0.05 -> water jug
```
When you submit query it will automatically download imagenet files, place then in `.spin/imagenet` sub-folder.

On my machine 
`spin/examples/http-rust-imagenet$ find .spin`

```
.spin
.spin/logs
.spin/logs/hello_stderr.txt
.spin/logs/hello_stdout.txt
.spin/imagenet
.spin/imagenet/model.xml
.spin/imagenet/model.bin
.spin/sqlite_key_value.db
```