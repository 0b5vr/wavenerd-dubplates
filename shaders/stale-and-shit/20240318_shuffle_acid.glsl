#define TRANSPOSE 3.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a, b, t) saturate((t - a) / (b - a))
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = 0; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p)-0.5))

const float SWING = 1.3;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);
const float MIN3 = pow(2.0, 3.0 / 12.0);
const float P4 = pow(2.0, 5.0 / 12.0);
const float P5 = pow(2.0, 7.0 / 12.0);

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

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
}

float p2f(float p) {
  return exp2((p - 69.0) / 12.0) * 440.0;
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

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

vec3 cyclicNoise(vec3 p, float pump) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  repeat(i, 5) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum *= pump;
    p *= 2.;
  }

  return sum.xyz / sum.w;
}

float cheapfiltersaw(float phase, float k) {
  float wave = mod(phase, 1.0);
  float c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c) * 2.0 - 1.0 - k;
}

vec2 ladderLPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;
  float b = 4.0 * omega * (omegaSq - 1.0);

  return vec2(
    1.0 / sqrt(a * a + b * b),
    atan(a, b)
  );
}

vec2 twoPoleHPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 2.0 * (1.0 - reso) * omega;
  float b = omegaSq - 1.0;

  return vec2(
    omegaSq / sqrt(a * a + b * b),
    atan(a, b)
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

vec4 seq16( int seq, float t, float tps ) {
  int sti = int( t / tps ) & 15;
  int rotated = ( ( seq >> ( 15 - sti ) ) | ( seq << ( sti + 1 ) ) ) & 0xffff;

  float prevStepBehind = log2( float( rotated & -rotated ) );
  float prevStep = float( sti ) - prevStepBehind;
  float nextStepForward = 16.0 - floor( log2( float( rotated ) ) );
  float nextStep = float( sti ) + nextStepForward;

  return vec4(
    prevStep,
    mod( t, 16.0 * tps ) - prevStep * tps,
    nextStep,
    nextStep * tps - mod( t, 16.0 * tps )
  );
}

vec4 quant(float x, float ks, float kt, out float i) {
  i = floor(floor(x / ks + 1E-4) * ks / kt + 1E-4);

  float s = kt <= ks
    ? ks * floor(x / ks + 1E-4)
    : ks * ceil(i * kt / ks - 1E-4);
  float l = kt <= ks
    ? ks
    : ks * ceil((i + 1.0) * kt / ks - 1E-4) - s;

  float t = x - s;
  float q = l - t;

  return vec4(s, t, s + l, q);
}

vec4 quant(float x, float ks, float kt) {
  float i;
  return quant(x, ks, kt, i);
}

float swing(float x, float k) {
  float xm = mod(x, 2.0);
  return x + (1.0 - k) * linearstep(0.0, k, xm) * linearstep(2.0, k, xm);
}

float unswing(float x0, float x, float y, float k) {
  return (
    x0
    - 2.0 * floor((x - y) / 2.0)
    - k * linearstep(0.0, 1.0, mod(x - y, 2.0))
    - (2.0 - k) * linearstep(1.0, 2.0, mod(x - y, 2.0))
  );
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float sidechain;

  { // kick
    float swinged = swing(time.y / S2T, SWING) * S2T;
    vec4 seq = seq16(0x889a, swinged, 0.25 * B2T);
    float t = unswing(time.y / S2T, swinged / S2T, seq.y / S2T, SWING) * S2T;
    float q = -unswing(time.y / S2T, swinged / S2T, -seq.w / S2T, SWING) * S2T;
    sidechain = 0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    float tt = t;
    float wave = sin(
      290.0 * tt
      - 40.0 * exp(-tt * 20.0)
      - 20.0 * exp(-tt * 40.0)
      - 20.0 * exp(-tt * 200.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // bass
    float swinged = swing(time.x / S2T, SWING) * S2T;
    float t = unswing(time.x / S2T, swinged / S2T, mod(swinged, S2T) / S2T, SWING) * S2T;
    float q = -unswing(time.x / S2T, swinged / S2T, -(1.0 - mod(swinged, S2T) / S2T), SWING) * S2T;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q - 0.01);

    float pitch = 24.0 + TRANSPOSE;
    float freq = p2f(pitch);
    vec2 fm = exp(-10.0 * t) * cis(2.08 * TAU * freq * t);
    vec2 wave = tanh(6.0 * exp(-10.0 * t) * sin(TAU * freq * t + fm));

    dest += 0.5 * env * sidechain * wave;
  }

  { // hihat
    float swinged = swing(time.y / S2T, SWING) * S2T;
    vec4 seq = quant(swinged, S2T, 1.42 * S2T);
    float t = unswing(time.y / S2T, swinged / S2T, seq.y / S2T, SWING) * S2T;
    float q = -unswing(time.y / S2T, swinged / S2T, -seq.w / S2T, SWING) * S2T;

    float env = exp(-20.0 * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.2 * env * tanh(8.0 * wave);
  }

  { // snare
    float t0 = mod(time.y - 2.0 * S2T, 4.0 * B2T);
    float swinged = swing(t0 / S2T, SWING) * S2T;
    vec4 seq = quant(swinged, S2T, 2.88 * S2T);
    float t = unswing(t0 / S2T, swinged / S2T, seq.y / S2T, SWING) * S2T;
    float q = -unswing(t0 / S2T, swinged / S2T, -seq.w / S2T, SWING) * S2T;

    float env = exp(-10.0 * t) * smoothstep(0.0, 0.001, q);

    float phase = 200.0 * t - 10.0 * exp(-400.0 * t);

    vec2 wave = vec2(0.0);
    wave += exp(-20.0 * t) * (
      + cis(TAU * phase)
      + cis(1.5 * TAU * phase)
    );
    wave += exp(-10.0 * t) * cheapnoise(50.0 * t);

    dest += 0.25 * env * tanh(8.0 * wave);
  }

  { // rim
    float swinged = swing(time.y / S2T, SWING) * S2T;
    float t = mod(mod(swinged, 5.0 * S2T), 2.0 * S2T);
    t = unswing(time.y / S2T, swinged / S2T, t / S2T, SWING) * S2T;

    float env = exp(-300.0 * t);

    float wave = tanh(4.0 * (
      +tri(t * 400.0 - 0.5 * env)
      +tri(t * 1500.0 - 0.5 * env)
    ));
    dest += 0.2 * env * wave * vec2(1, -1);
  }

  { // crash
    float t = time.z;

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 3.0);
    dest += 0.3 * env * sidechain * tanh(8.0 * wave);
  }

  { // acid
    const int N_NOTES = 5;
    const int NOTES[N_NOTES] = int[](0, 12, 13, 10, 7);

    float t0 = mod(time.z, 16.0 * B2T);
    float swinged = swing(t0 / S2T, SWING) * S2T;
    float seqi;
    vec4 seq = quant(swinged, S2T, 1.22 * S2T, seqi);
    float t = unswing(t0 / S2T, swinged / S2T, seq.y / S2T, SWING) * S2T;
    float q = -unswing(t0 / S2T, swinged / S2T, -seq.w / S2T, SWING) * S2T;

    vec2 sum = vec2(0.0);

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q - 0.02);

    float cutoff = exp2(8.0 + 3.0 * smoothstep(0.0, 0.02, t) * exp(-8.0 * t));
    float reso = 0.7;

    repeat(i, 128) {
      float fi = float(i);

      float pitch = 36.0 + TRANSPOSE + float(NOTES[int(seqi) % N_NOTES]);

      float p = 1.0 + fi;
      float freq = p2f(pitch) * p;

      vec2 filt = ladderLPF(freq, cutoff, reso);
      float phase = t * freq;

      vec2 wave = vec2(0.0);
      wave += cis(TAU * phase + filt.y);
      sum += wave * env * filt.x / p * rotate2D(2.4 * fi + time.w);
    }

    dest += 0.2 * tanh(10.0 * sum);
  }

  return tanh(dest);
}
