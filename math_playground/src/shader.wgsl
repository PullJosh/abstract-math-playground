// Vertex shader
struct TimeUniform {
    time: f32,
};

@group(0) @binding(0)
var<uniform> time: TimeUniform;

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @builtin(vertex_index) index: u32,
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.color;
    return vec4<f32>(color, 1.0);
}

// WebGPU hsv to rgb conversion
// All components are in the range [0…1], including hue.
fn hsv2rgb(rgb: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(rgb.xxx + K.xyz) * 6.0 - K.www);
    return rgb.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), rgb.y);
}