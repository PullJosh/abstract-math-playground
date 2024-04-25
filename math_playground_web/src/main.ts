export {}; // This file is a module (allows for top-level await)

const MathPlaygroundLib = await import(
  "../../math_playground/pkg/math_playground_lib"
);

try {
  MathPlaygroundLib.run();
} catch (err) {
  const supress =
    err instanceof Error &&
    err.message ===
      "Using exceptions for control flow, don't mind me. This isn't actually an error!";

  if (!supress) {
    throw err;
  }
}
