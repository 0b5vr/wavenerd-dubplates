#pragma use_binary_literals

#define TRANSPOSE 3.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float SWING = 0.54;

const float LN2 = log(2.0);
const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float P5 = pow(2.0, 7.0 / 12.0);

uniform vec4 param_knob4; // acid cutoff
uniform vec4 param_knob5; // acid reso
uniform vec4 param_knob6; // acid centroid
uniform vec4 param_knob7; // kick cut

#define p4 paramFetch(param_knob4)
#define p5 paramFetch(param_knob5)
#define p6 paramFetch(param_knob6)
#define p7 paramFetch(param_knob7)

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

vec4 quant(float t, float interval, out float i) {
  interval = max(interval, 1.0);
  float st = t2sSwing(t);

  i = floor(floor(st) / interval);

  float prevStep = ceil(i * interval);
  float prevTime = s2tSwing(prevStep);
  float nextStep = ceil((i + 1.0) * interval);
  float nextTime = s2tSwing(nextStep);

  return vec4(
    prevStep,
    t - prevTime,
    nextStep,
    nextTime - t
  );
}

vec4 quant(float t, float interval) {
  float _;
  return quant(t, interval, _);
}

float p2f(float p) {
  return exp2((p - 69.0) / 12.0) * 440.0;
}

vec2 ladderLPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 4.0 * omega * (omegaSq - 1.0);
  float b = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;

  return vec2(
    1.0 / sqrt(a * a + b * b),
    atan(a, b)
  );
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

float glidephase(float t, float t1, float p0, float p1) {
  if (p0 == p1 || t1 == 0.0) {
    return t * p2f(p1);
  }

  float m0 = (p0 - 69.0) / 12.0;
  float m1 = (p1 - 69.0) / 12.0;
  float b = (m1 - m0) / t1;

  return (
    + p2f(p0) * (
      + min(t, 0.0)
      + (pow(2.0, b * clamp(t, 0.0, t1)) - 1.0) / b / LN2
    )
    + max(0.0, t - t1) * p2f(p1)
  );
}

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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float duck = smoothstep(0.0, 0.4, time.x) * smoothstep(0.0, 0.001, B2T - time.x);

  { // kick
    vec4 seq = seq16(time.y, 0b1000100010001000);
    float t = seq.t;
    float q = seq.q;
    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    float env = smoothstep(0.0, 0.001, q) * exp(-20.0 * max(t - 0.1, 0.0));
    env *= mix(1.0, exp(-50.0 * t), p7);

    // {
    //   float tt = t;
    //   float wave = 0.0;

    //   wave += tanh(2.5 * sin(TAU * (
    //     50.0 * tt
    //     - 8.0 * exp2(-tt * 33.0)
    //   )));

    //   float tick = env * exp2(-500.0 * t);
    //   wave += env * tanh(1.5 * sin(TAU * 3.0 * tick));

    //   dest += 0.6 * env * wave;
    // }
  }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0b1111111111111111);
  //   float t = seq.y;

  //   float envseq[] = float[](
  //     0.0, 0.0, 0.0, 0.0,
  //     0.0, 0.0, 0.0, 1.0,
  //     0.0, 1.0, 0.0, 0.0,
  //     0.0, 0.3, 0.5, 1.0
  //   );
  //   float kenv = exp2(mix(7.5, 5.0, envseq[int(seq.s)]));
  //   float env = exp2(-kenv * t);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 4.5 * exp2(-10.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * t + dice2.xy);
  //     wave = 3.2 * exp2(-10.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * exp2(-10.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.2 * env * mix(0.5, 1.0, duck) * tanh(2.0 * sum);
  // }

  // { // clap
  //   vec4 seq = seq16(time.y, 0b0010000000000001);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = mix(
  //     exp2(-40.0 * t),
  //     exp2(-300.0 * mod(t, 0.01)),
  //     exp2(-100.0 * max(0.0, t - 0.02))
  //   );

  //   float radius = 2.0 + 2.0 * exp(-20.0 * t);
  //   float phase = 420.0 * t;
  //   vec2 wave = cyclic(vec3(radius * cis(TAU * phase), TAU * phase), 0.6, 2.0).xy;

  //   dest += 0.2 * mix(0.8, 1.0, duck) * tanh(20.0 * env * wave);
  // }

  // { // snare909
  //   vec4 seq = seq16(time.y, 0b0010010101000011);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = exp(-20.0 * max(t - 0.04, 0.0)) * smoothstep(0.0, 0.01, q);

  //   float noisephase = 600.0 * t;
  //   vec2 wave = mix(
  //     cis(TAU * (220.0 * t - 4.0 * exp2(-t * 200.0))),
  //     cheapnoise(128.0 * t) - cheapnoise(128.0 * t - 0.008),
  //     0.3
  //   );

  //   dest += 0.3 * mix(0.5, 1.0, duck) * tanh(4.0 * env * wave);
  // }

  // { // hi tom
  //   vec4 seq = seq16(time.y, 0b0001000001010000);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = exp(-20.0 * t);
  //   float freq = 110.0;
  //   float phase = (
  //     t
  //     - 0.03 * exp2(-40.0 * t)
  //     - 0.01 * exp2(-150.0 * t)
  //   );
  //   phase *= TAU * freq;

  //   vec2 wave = cis(phase + sin(3.0 * phase) + 10.0 * t);
  //   wave.x *= 0.5;

  //   dest += 0.2 * mix(0.8, 1.0, duck) * tanh(2.0 * env * wave);
  // }

  // { // low tom
  //   vec4 seq = seq16(time.y, 0b0000001000000010);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = exp(-20.0 * t);
  //   float freq = 80.0;
  //   float phase = (
  //     t
  //     - 0.03 * exp2(-40.0 * t)
  //     - 0.01 * exp2(-150.0 * t)
  //   );
  //   phase *= TAU * freq;

  //   vec2 wave = cis(phase + sin(3.0 * phase) + 10.0 * t);
  //   wave.y *= 0.5;

  //   dest += 0.2 * mix(0.8, 1.0, duck) * tanh(2.0 * env * wave);
  // }

  // { // rim
  //   vec4 seq = seq16(time.y, 0b1101011011010111);
  //   float t = seq.y;

  //   float env = step(0.0, t) * exp2(-400.0 * t);

  //   float wave = tanh(4.0 * (
  //     + tri(t * 400.0 - 0.5 * env)
  //     + tri(t * 1500.0 - 0.5 * env)
  //   ));

  //   dest += 0.2 * mix(0.8, 1.0, duck) * env * vec2(wave) * rotate2D(seq.x);
  // }

  // { // ride
  //   float t = mod(time.x - 0.5 * B2T, 1.0 * B2T);

  //   float env = exp(-2.0 * t);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 2.5 * env * sin(wave + exp2(14.90 + 0.1 * dice.x) * t + dice2.xy);
  //     wave = 4.2 * env * sin(wave + exp2(13.27 + 0.5 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * env * sin(wave + exp2(13.89 + 1.0 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.15 * mix(0.2, 1.0, duck) * env * tanh(sum);
  // }

  // { // crash
  //   float t = mod(time.z, 64.0 * B2T);

  //   float env = mix(exp(-t), exp(-10.0 * t), 0.7);
  //   vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 1.0);
  //   dest += 0.4 * env * mix(0.5, 1.0, duck) * tanh(8.0 * wave);
  // }

  { // acid
    const int N_NOTES = 5;
    const int NOTES[N_NOTES] = int[](0, 12, 18, 3, 9);
    const int SLIDE[N_NOTES] = int[](0, -12, 0, 12, -12);
    float SLIDE_T0 = 0.8 * S2T;
    float SLIDE_TIME = 0.6 * S2T;

    float seqi;
    vec4 seq = quant(time.z, 1.15, seqi);
    float t = seq.t;
    float q = seq.q;

    q -= mix(0.01, 0.15 * B2T, fract(seqi * 0.389));
    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);

    float cenv = smoothstep(0.0, 0.01, t) * exp(-8.0 * t);
    float cutoff = (
      8.0
      + exp2(mix(-2.0, 2.0, fract(seqi * 0.612))) * cenv
      + 3.0 * paramFetch(param_knob4)
    );
    float cfreq = exp2(cutoff);
    float reso = paramFetch(param_knob5);

    int i = int(seqi) % N_NOTES;
    float pitch = 36.0 + TRANSPOSE + float(NOTES[i]);
    float pitch1 = pitch + float(SLIDE[i]);
    float basefreq = p2f(mix(pitch, pitch1, linearstep(0.0, SLIDE_TIME, t - SLIDE_T0)));
    float basephase = glidephase(t - SLIDE_T0, SLIDE_TIME, pitch, pitch1);

    vec2 sum = vec2(0.0);

    { // sub
      float phase = 0.5 * basephase;
      dest += 0.3 * env * tanh(1.5 * sin(TAU * phase));
    }

    repeat(i, 128) { // acid
      float fi = float(i);
      vec3 dice = hash3f(vec3(i) + vec3(1, 5, 7));

      float p = 1.0 + 1.0 * fi;
      p = mix(p, 8.0, p6);
      float freq = basefreq * p;
      float coeff = exp(-0.1 * p);

      vec2 filt = ladderLPF(freq, cfreq, reso);
      float phase = basephase * p;
      // phase += TAU * dice.z;

      vec2 wave = vec2(0.0);
      wave += sin(TAU * phase + filt.y);
      sum += wave * env * coeff * filt.x;
    }

    dest += 0.25 * mix(0.8, 1.0, duck) * clip(4.0 * sum);
  }

  return clip(1.3 * tanh(dest));
}
