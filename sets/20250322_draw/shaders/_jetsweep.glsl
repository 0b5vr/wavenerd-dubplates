#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float SWING = 0.5;

const float LN2 = log(2.0);
const float PI = acos(-1.0);
const float TAU = PI * 2.0;

uniform vec4 param_knob0; // duck

#define p0 paramFetch(param_knob0)

uvec3 hash3u(uvec3 v) {
  v = v * 1145141919u + 1919810u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

vec3 hash3f(vec3 v) {
  uvec3 x = floatBitsToUint(v);
  return vec3(hash3u(x)) / float(-1u);
}

float tmod(vec4 time, float d) {
  vec4 t = mod(time, timeLength);
  float offset = lofi(t.z - t.x + timeLength.x / 2.0, timeLength.x);
  offset -= lofi(t.z, d);
  return t.x + offset;
}

vec2 cheapnoise(float t) {
  uvec3 s=uvec3(t * 256.0);
  float p=fract(t * 256.0);

  vec3 dice;
  vec2 v = vec2(0.0);

  dice=vec3(hash3u(s + 0u)) / float(-1u) - vec3(0.5, 0.5, 0.0);
  v += dice.xy * smoothstep(1.0, 0.0, abs(p + dice.z));
  dice=vec3(hash3u(s + 1u)) / float(-1u) - vec3(0.5, 0.5, 1.0);
  v += dice.xy * smoothstep(1.0, 0.0, abs(p + dice.z));
  dice=vec3(hash3u(s + 2u)) / float(-1u) - vec3(0.5, 0.5, 2.0);
  v += dice.xy * smoothstep(1.0, 0.0, abs(p + dice.z));

  return 2.0 * v;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float duck = smoothstep(0.0, B2T, time.x) * smoothstep(0.0, 0.001, B2T - time.x);

  if (time.z > 32.0 * B2T) { // sweep
    float t = tmod(time, 32.0 * B2T);

    float env = smoothstep(0.0, 16.0 * B2T, t);

    vec2 wave = vec2(0.0);
    wave += cheapnoise(128.0 * t);
    wave += cheapnoise(128.0 * (t + 0.002 * exp(-0.15 * t)));
    wave += cheapnoise(128.0 * (t + 0.004 * exp(-0.15 * t)));

    dest += 0.5 * mix(p0, 1.0, duck) * env * wave;
  }

  return clip(1.3 * tanh(dest));
}
