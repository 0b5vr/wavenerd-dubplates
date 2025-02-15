#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(x) clamp(x, -1., 1.)
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

vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D(float x) {
  vec2 v = cis(x);
  return mat2(v.x, v.y, -v.y, v.x);
}

float t2sSwing(float t) {
  float st = 4.0 * t / B2T;
  return 2.0 * floor(st / 2.0) + step(SWING, fract(0.5 * st));
}

float s2tSwing(float st) {
  return 0.5 * B2T * (floor(st / 2.0) + SWING * mod(st, 2.0));
}

vec4 seq16(float t, int seq) {
  t = mod(t, 4.0 * B2T);
  int sti = clamp(int(t2sSwing(t)), 0, 15);
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float i_prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - i_prevStepBehind;
  float prevTime = s2tSwing(prevStep);
  float i_nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + i_nextStepForward;
  float nextTime = s2tSwing(nextStep);

  return vec4(
    prevStep,
    t - prevTime,
    nextStep,
    nextTime - t
  );
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

  { // snare909
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    if (time.z > 60.0 * B2T) {
      float l = 0.125 * B2T;
      t = mod(time.x, l);
      q = l - t;
    }

    float env = smoothstep(0.0, 0.01, q);
    env *= mix(
      exp(-20.0 * max(t - 0.04, 0.0)),
      exp(-80.0 * t),
      0.3
    );

    float sinphase = 220.0 * t - 4.0 * exp2(-t * 200.0);
    float noisephase = 128.0 * t;
    vec2 wave = mix(
      mix(
        cis(TAU * (sinphase)),
        cis(TAU * (1.5 * sinphase)),
        0.3
      ),
      cheapnoise(noisephase) - cheapnoise(noisephase - 0.004),
      0.4
    );

    dest += 0.3 * mix(p0, 1.0, duck) * tanh(3.0 * env * wave);
  }

  return clip(1.3 * tanh(dest));
}
