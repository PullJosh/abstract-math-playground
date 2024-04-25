# Abstract Math Playground

Right now this project does not have a clear purpose or goal except to be a playground where I explore what makes me curious. It lives in the same realm as [Mr. Pullen's Graphing Calculator](https://calculator.mrpullen.com/). If you're interested in these kinds of things and want to discuss more, join us on Discord!

[![Join the Graphing Calculator Creators Discord server](https://invidget.switchblade.xyz/sgqwmkUQhQ)](https://discord.gg/sgqwmkUQhQ)

# Installation & Usage

To be honest, I don't remember all the exact installation steps. If someone installs this and can write down what the steps should be, that would be lovely.

Generally speaking, you want to do this:

## Install dependencies

- Install rust/cargo on your machine (I used rustup)
- Install node/npm (if you want to run the web version as well)

## Run the native version

```bash
cd math_playground
cargo run
```

## Run the web version

```bash
cd math_playground_web
npm run wasm-dev # Builds the wasm/webgpu code from `math_playground` once, then runs the vite dev server
```
