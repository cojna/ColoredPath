#!/bin/bash -e

cargo build --release

for file in examples/*; do
    cargo run --release < $file > /dev/null
done
