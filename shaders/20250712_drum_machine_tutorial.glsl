#pragma use_binary_literals

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define u2b(u) ((u) * 2.0 - 1.0)
#define b2u(b) ((b) * 0.5 + 0.5)
#define tri(x) (1.0 - 4.0 * abs(fract((x) + 0.25) - 0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

const float SWING = 0.54;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;

// == hash stuff ===================================================================================
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

// == math utils ==
vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D(float t) {
  float c = cos(t);
  float s = sin(t);
  return mat2(c, s, -s, c);
}

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
}

// == sequencer functions ==========================================================================
float tmod(vec4 time, float d) {
  vec4 t = mod(time, timeLength);
  float offset = lofi(t.z - t.x + timeLength.x / 2.0, timeLength.x);
  offset -= lofi(t.z, d);
  return t.x + offset;
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

// == noise generators =============================================================================
mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

vec3 cyclic(vec3 p, float pers, float lacu) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  repeat(i, 5) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum /= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
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

vec2 shotgun(float t, float spread, float snap, float fm) {
  vec2 sum = vec2(0.0);

  repeat(i, 64) {
    vec3 dice = hash3f(vec3(i + 1));

    vec2 partial = exp2(spread * dice.xy);
    partial = mix(partial, floor(partial + 0.5), snap);

    sum += sin(TAU * t * partial + fm * sin(TAU * t * partial));
  }

  return sum / 64.0;
}

// == drums ========================================================================================
vec2 kick(float t, float q) {
  // envelope
  float env = smoothstep(0.0, 0.001, q);
  env *= smoothstep(0.3, 0.1, t);

  // phase - 50Hz + fall
  float phase = 50.0 * t;
  phase += 8.0 * (1.0 - exp2(-50.0 * t));

  // oscillator - simple sinewave
  vec2 osc = vec2(sin(TAU * phase));

  // transient
  float tphase = 5.0 * (1.0 - exp2(-400.0 * t));
  osc += sin(TAU * tphase);

  // add overdrive to osc
  osc = tanh(osc);

  return env * osc;
}

vec2 hihat(float t, float q) {
  // envelope - simple exponential decay
  float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  env *= exp2(-70.0 * t);

  // oscillator - sinewave shotgun
  vec2 osc = shotgun(5000.0 * t, 2.0, 0.0, 0.0);
  osc = tanh(4.0 * osc);

  return env * osc;
}

vec2 snare(float t, float q) {
  // envelope - exponential decay with initial hold
  float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  env *= exp(-20.0 * max(t - 0.04, 0.0));

  // phase for body - 220Hz + fall
  float phase = 220.0 * t;
  phase += 4.0 * (1.0 - exp2(-t * 200.0));

  // oscillator - two sinewaves + noise
  vec2 osc = mix(
    mix(
      cis(TAU * phase), // 1x freq sine
      cis(1.5 * TAU * phase), // 1.5x freq sine
      0.4
    ),
    cheapnoise(128.0 * t) - cheapnoise(128.0 * t - 0.008), // noise
    0.3
  );

  return env * osc;
}

vec2 clap(float t, float q) {
  // envelope - mixing exponential decay and exponential decay with retrigger
  float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  env *= mix(
    exp2(-40.0 * t),
    exp2(-300.0 * mod(t, 0.012)),
    exp2(-100.0 * max(0.0, t - 0.02))
  );

  // oscillator - cyclic noise
  vec2 osc = cyclic(vec3(4.0 * cis(1200.0 * t), 1940.0 * t), 0.5, 2.0).xy;

  return env * osc;
}

vec2 ride(float t, float q) {
  // envelope - simple exponential decay
  float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  env *= exp2(-2.0 * t);

  // oscillator - unison random fm sinewaves
  vec2 sum = vec2(0.0);
  repeat(i, 8) {
    vec3 dice = hash3f(vec3(i));
    vec3 dice2 = hash3f(dice);

    vec2 osc = vec2(0.0);
    osc = 2.5 * env * sin(osc + exp2(14.90 + 0.1 * dice.x) * t + dice2.xy);
    osc = 4.2 * env * sin(osc + exp2(13.27 + 0.5 * dice.y) * t + dice2.yz);
    osc = 1.0 * env * sin(osc + exp2(13.89 + 1.0 * dice.z) * t + dice2.zx);

    sum += osc / 4.0;
  }
  sum = tanh(sum);

  return env * sum;
}

vec2 tom(float t, float q, float freq) {
  // envelope - simple exponential decay
  float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  env *= exp2(-20.0 * t);

  // phase - given frequency + fall
  float phase = t;
  phase += 0.03 * (1.0 - exp2(-40.0 * t));
  phase += 0.01 * (1.0 - exp2(-150.0 * t));
  phase *= freq;

  // oscillator - sinewave with fm
  vec2 osc = vec2(sin(TAU * phase + 0.5 * sin(3.0 * TAU * phase)));

  return env * osc;
}

vec2 rim(float t, float q) {
  // envelope - simple exponential decay
  float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  env *= exp2(-400.0 * t);

  // oscillator - two triwaves
  vec2 osc = vec2(tanh(4.0 * (
    + tri(t * 400.0 - 0.5 * env)
    + tri(t * 1500.0 - 0.5 * env)
  )));

  return env * osc;
}

vec2 clav(float t, float q) {
  // envelope - simple exponential decay
  float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  env *= exp2(-100.0 * t);

  // oscillator - triwave
  vec2 osc = vec2(tri(5000.0 * t));

  return env * osc;
}

vec2 crash(float t, float q) {
  // envelope - mix two exponential decays
  float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  env *= mix(
    exp2(-t),
    exp2(-10.0 * t),
    0.5
  );

  // oscillator - sinewave shotgun
  vec2 osc = shotgun(3800.0 * t, 2.0, 0.0, 1.0);
  osc = tanh(4.0 * osc);

  return env * osc;
}

// == main =========================================================================================
vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  // sidechain / ducking
  float duck = 1.0;

  { // kick
    vec4 seq = seq16(time.y, 0b1000100010001000);
    duck = smoothstep(0.0, 0.3, seq.t) * smoothstep(0.0, 0.001, seq.q);
    dest += 0.5 * kick(seq.t, seq.q);
  }

  { // hihat
    vec4 seq = seq16(time.y, 0b1111111111111111);
    dest += 0.2 * mix(0.2, 1.0, duck) * hihat(seq.t, seq.q);
  }

  { // snare
    vec4 seq = seq16(time.y, 0b0011000100110110);
    dest += 0.3 * mix(0.2, 1.0, duck) * snare(seq.t, seq.q);
  }

  { // clap
    vec4 seq = seq16(time.y, 0b0010000001000010);
    dest += 0.4 * mix(0.3, 1.0, duck) * clap(seq.t, seq.q);
  }

  { // ride
    vec4 seq = seq16(time.y, 0b1010101010101010);
    dest += 0.03 * mix(0.1, 1.0, duck) * ride(seq.t, seq.q);
  }

  { // tom (hi)
    vec4 seq = seq16(time.y, 0b0001000000010000);
    dest += vec2(0.3, 0.1) * mix(0.3, 1.0, duck) * tom(seq.t, seq.q, 110.0);
  }

  { // tom (lo)
    vec4 seq = seq16(time.y, 0b0000001000000010);
    dest += vec2(0.1, 0.3) * mix(0.3, 1.0, duck) * tom(seq.t, seq.q, 80.0);
  }

  { // rim
    vec4 seq = seq16(time.y, 0b0110110101110101);
    dest += vec2(0.2, 0.1) * mix(0.3, 1.0, duck) * rim(seq.t, seq.q);
  }

  { // clav
    vec4 seq = seq16(time.y, 0b1010100101101011);
    dest += vec2(0.1, 0.2) * mix(0.3, 1.0, duck) * clav(seq.t, seq.q);
  }

  { // crash
    float t = time.z;
    float q = 64.0 * B2T - time.z;
    dest += 0.4 * mix(0.2, 1.0, duck) * crash(t, q);
  }

  { // bonus: bass
    float t = time.x;
    float q = B2T - time.x;

    // envelope
    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

    // pitch, frequency, and phase
    float pitch = time.y < 3.0 * B2T ? 30.0 : 40.0;
    float freq = p2f(pitch);
    float phase = freq * t;

    // add phase modulation to phase
    float fmamp = 0.2 * smoothstep(0.0, B2T, t);
    phase += fmamp * sin(2.0 * TAU * phase);

    // oscillator - sinewave
    float osc = sin(TAU * phase);

    // add overdrive to osc
    osc = tanh(4.0 * osc);

    dest += 0.2 * mix(0.0, 1.0, duck) * env * osc;
  }

  // master dynamics - clip + tanh
  return clip(1.3 * tanh(dest));
}
