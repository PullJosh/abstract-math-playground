// Vertex shader
struct TimeUniform {
    time: f32,
};

@group(0) @binding(0)
var<uniform> time: TimeUniform;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Make an equilateral triangle
    let angle = f32(in_vertex_index) * radians(120.0) + time.time * radians(180.0);
    let x = cos(angle) * 0.5;
    let y = sin(angle) * 0.5;

    let dir = 0.25 * time.time * radians(360.0);
    let len = sin(NUMBER * dir + radians(180.0)) * 0.25;
    out.vert_pos = vec3<f32>(x, y, 0.0) + len * vec3<f32>(cos(dir), sin(dir), 0.0);
    out.clip_pos = vec4<f32>(out.vert_pos, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pos = in.vert_pos.xy * 0.5 + vec2<f32>(0.5, 0.5);
    let dist = length(pos - vec2<f32>(0.5, 0.5));
    let angle = atan2(pos.y - 0.5, pos.x - 0.5);
    let hue = angle / radians(360.0) + pow(2.0, dist * 5.0) - time.time;
    let color = hsv2rgb(vec3<f32>(hue, 1.0, 1.0));
    return vec4<f32>(color, 1.0);
}

// WebGPU hsv to rgb conversion
// All components are in the range [0…1], including hue.
fn hsv2rgb(rgb: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(rgb.xxx + K.xyz) * 6.0 - K.www);
    return rgb.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), rgb.y);
}