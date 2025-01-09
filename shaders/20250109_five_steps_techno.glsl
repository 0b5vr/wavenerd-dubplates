#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(i) clamp(i, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p-0.25)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

const float SWING = 0.51;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);
const float MIN3 = pow(2.0, 3.0 / 12.0);
const float P4 = pow(2.0, 5.0 / 12.0);
const float P5 = pow(2.0, 7.0 / 12.0);

uniform vec4 param_knob0;

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

  for (int i = 0; i ++ < 5;) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum /= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
}

float cheapfiltersaw(float phase, float k) {
  float wave = fract(phase);
  float c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
}

vec2 cheapfiltersaw(vec2 phase, float k) {
  vec2 wave = fract(phase);
  vec2 c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
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

vec2 mainAudioDry(vec4 time) {
  vec2 dest = vec2(0);
  float duck = 1.0;

  const int N_CHORD = 8;
  int CHORD[] = int[](
    7, 10, 12, 14, 15, 17, 19, 22
  );

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);
    duck = pow(duck, 2.0);

    float env = smoothstep(0.0, 0.001, q) * exp2(-20.0 * max(t - 0.1, 0.0));

    // {
    //   env *= exp2(-90.0 * t);
    // }

    {
      float phase = (
        42.0 * t
        - 4.0 * exp2(-t * 30.0)
        - 4.0 * exp2(-t * 70.0)
      );
      float wave = sin(TAU * phase + 0.3 * cos(TAU * phase));
      dest += 0.6 * clip(3.0 * env * wave);
    }
  }

  { // bass
    float t = time.x;
    float q = B2T - t;

    float env = smoothstep(0.0, 0.004, t) * smoothstep(0.0, 0.004, q);

    float p0 = 32.0 + TRANSPOSE;
    float p1 = p0 + mix(-2.0, 2.0, step(B2T, mod(time.y, 2.0 * B2T)));
    float phase = glidephase(t, B2T, p0, p1);
    float wave = tanh(2.0 * sin(TAU * phase));

    dest += 0.4 * mix(0.0, 1.0, duck) * env * wave;
  }

  { // hihat
    const int i_patternCH = 0xffff;
    vec4 seq = seq16(time.y, i_patternCH);
    float t = seq.y;

    float open = float(mod(seq.x, 4.0) == 2.0);
    float env = exp2(-exp2(6.0 - 2.0 * open) * t);

    vec2 wave = shotgun(7000.0 * t, 1.4, 0.0, 0.0);
    dest += 0.16 * env * mix(0.2, 1.0, duck) * tanh(8.0 * wave);
  }

  { // noise perc
    vec4 seq = seq16(time.y, 0x6d6d);
    float t = seq.t;
    float q = seq.q;
    float s = seq.s;

    float env = smoothstep(0.02, 0.0, t - 0.07);

    vec3 p = vec3(cis(3000.0 * t), 280.0 * t);
    float pers = 0.6;
    float lacu = 1.8;
    vec2 wave = cyclic(p, pers, lacu).xy;

    dest += 0.3 * env * mix(0.3, 1.0, duck) * wave;
  }

  { // click perc
    vec4 seq = seq16(time.y, 0x576f);
    float t = seq.t;
    float q = seq.q;
    float s = seq.s;

    float env = exp2(-200.0 * t);

    float freq = exp2(lofi(10.5 + sin(2.4 * s), 0.7));
    float phase = freq * t;

    vec2 wave = vec2(tanh(sin(TAU * phase)));
    wave = tanh(8.0 * wave);
    wave *= rotate2D(s);
    dest += 0.12 * env * mix(0.2, 1.0, duck) * wave;
  }

  { // five steps osc
    int N_STEPS = 5;
    float SEQ[] = float[](0.08, 0.34, 0.59, 0.91, 0.48);

    vec2 sum = vec2(0.0);
    repeat(iDelay, 5) {
      float fiDelay = float(iDelay);
      float delaydecay = exp2(-0.4 * fiDelay - 1.0 * step(1.0, fiDelay));
      float offset = -2.4 * S2T * fiDelay;

      vec4 seq = seq16(time.y + offset, 0xffff);
      float t = seq.t;
      float q = seq.q;
      float st = seq.s + 16.0 * floor(tmod(time + offset, timeLength.z) / (4.0 * B2T));
      int ip = int(mod(st, float(N_STEPS)));

      vec3 dice = hash3f(vec3(st, 21, 28));

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      float rise = 0.1 * S2T;
      float fall = 1.6 * S2T;
      float menv = linearstep(0.0, rise, t) * linearstep(fall + rise, rise, t);
      // env *= menv;

      float pa = 24.0;
      float pb = 48.0 + TRANSPOSE;
      float p0 = pb + pa * float(SEQ[(ip + N_STEPS - 1) % N_STEPS]);
      float p1 = pb + pa * float(SEQ[ip]);
      vec2 phase = glidephase(t, 0.01, p0, p1) * vec2(0.99, 1.01);

      float kk = (
        + 1.0
        + 3.0 * menv
        + 0.5 * sin(time.w / 2.0)
        + 1.0 * dice.x
        - 0.5 * fiDelay
      );
      float k = saturate(1.0 - exp2(-kk));
      vec2 wave = vec2(cheapfiltersaw(phase, k));
      wave = sin(exp2(4.0 * paramFetch(param_knob0)) * wave);
      wave *= rotate2D(st);
      sum += env * wave * delaydecay;
    }

    dest += 0.3 * mix(0.3, 1.0, duck) * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  return clip(1.3 * tanh(dest));
}
