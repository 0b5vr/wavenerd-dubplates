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
#define TRANSPOSE 3.0

const float SWING = 0.5;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

uniform vec4 param_knob0; // pad level
uniform vec4 param_knob3; // kick cut

#define p0 paramFetch(param_knob0)
#define p3 paramFetch(param_knob3)

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

float s2tSwing(int st) {
  return s2tSwing(float(st));
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

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

float glidephase(float t, float t1, float pitch0, float pitch1) {
  if (pitch0 == pitch1) {
    return t * p2f(pitch1);
  }

  float m0 = (pitch0 - 69.0) / 12.0;
  float m1 = (pitch1 - 69.0) / 12.0;
  float b = (m1 - m0) / t1;

  return (
    + p2f(pitch0) * (pow(2.0, b * min(t, t1)) - 1.0) / b / LN2
    + max(0.0, t - t1) * p2f(pitch1)
  );
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

vec3 corruptor(float t, float corrupt) {
  const float PERS = 0.5;
  const float LACU = 1.4;

  float n = cyclic(vec3(t, corrupt, 0.0), PERS, LACU).x;
  n = floor(n + corrupt);
  return hash3f(vec3(t, 0.0, n));
}

float cheapfiltersaw(float phase, float k) {
  float wave = fract(phase);
  float c = smoothstep(1.0, 0.0, wave / (1.0 - k));
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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float duck = 1.0;

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    {
      float env = smoothstep(0.0, 0.001, q);
      env *= smoothstep(0.3, 0.1, t);
      env *= mix(1.0, exp(-50.0 * t), p3);

      float phase = (
        40.0 * t
        - 2.0 * exp2(-t * 20.0)
        - 1.0 * exp2(-t * 60.0)
        - 1.0 * exp2(-t * 100.0)
      );
      phase += 0.1 * exp2(-40.0 * t) * sin(5.0 * TAU * phase); // fm attack

      float wave = tanh(sin(TAU * phase));

      dest += 0.8 * env * wave;
    }
  }

  { // bass
    float l = B2T;
    float t = time.x;
    float q = l - t;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    float progtrans = mod(time.z / S2T, 128.0) < 64.0 ? 0.0 : -3.0;
    float pitch = 24.0 + TRANSPOSE + progtrans;
    float freq = p2f(pitch);
    float phase = freq * t;

    vec2 osc = vec2(0.0);
    osc += sin(TAU * phase);

    dest += 0.5 * mix(0.0, 1.0, duck) * env * osc;
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp2(-100.0 * t);

    vec2 wave = shotgun(3700.0 * t, 2.0, 0.0, 0.0);
    wave = tanh(2.0 * wave);

    dest += 0.3 * mix(0.1, 1.0, duck) * env * wave;
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp2(-60.0 * max(t - 0.05, 0.0));

    vec2 wave = shotgun(2400.0 * t, 2.2, 0.0, 2.0);
    wave = tanh(2.0 * wave);

    dest += 0.6 * mix(0.1, 1.0, duck) * env * wave;
  }

  { // rim
    float t = min(
      quant(time.z, 3.21).t,
      quant(time.z, 2.44).t
    );

    float env = exp2(-400.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * mix(0.3, 1.0, duck) * vec2(wave) * rotate2D(time.z - t);
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4500.0 * t, 1.4, 0.0, 1.0);
    dest += 0.4 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  const int N_CHORD = 8;
  int CHORD[] = int[](
    0, 7, 10, 14, 15, 17, 19, 22
  );

  { // chord
    vec4 seqg = seq16(time.y, 0xffff);
    float gate = (
      smoothstep(0.0, 0.001, seqg.t)
      * smoothstep(0.0, 0.001, seqg.q)
      * exp2(-50.0 * max(0.4 * S2T - seqg.q, 0.0))
    );

    vec2 sum = vec2(0.0);
    repeat(iUnison, 128) {
      vec3 dice = hash3f(vec3(iUnison, 7, 7));
      vec2 dicen = boxMuller(dice.xy);

      vec4 tdelay = mod(time - 0.5 * dice.y, timeLength);
      float l = 16.0 * B2T;
      float t = mod(tdelay.z, l);
      float st = round((tdelay.z - t) / S2T);

      float fade = smoothstep(0.0, S2T, t) * smoothstep(0.0, S2T, l - t);

      float progtrans = mod((tdelay.z) / l, 2.0) < 1.0 ? 0.0 : -3.0;
      float pitch = 48.0 + TRANSPOSE + progtrans + float(CHORD[iUnison % N_CHORD]);
      float freq = p2f(pitch) * exp2(0.01 * dicen.x);

      float phase = t * freq + dice.x;

      vec3 p = vec3(0.5 * cis(TAU * phase) + 7.0, 4.0);
      p += 0.4 * cos(TAU * time.z / B2T / 32.0);

      vec2 wave = cyclic(p, 0.01 + p0 * 0.5, 2.0).xy;

      sum += fade * wave * rotate2D(2.4 * float(iUnison));
    }

    dest += 0.03 * p0 * sum * mix(0.0, 1.0, gate) * mix(0.2, 1.0, duck);
  }

  { // plucks
    vec2 sum = vec2(0.0);
    repeat(i, 128) {
      vec3 dice = hash3f(vec3(i / 4, 12, 18));
      vec3 dicetime = hash3f(vec3(i / 4, 50, 54));

      float delay = float(i % 4);
      float delaydecay = exp2(-delay);

      vec4 tdelay = mod(time - delay * 3.0 * S2T, timeLength);
      float l = S2T * floor(3.0 + 9.0 * dicetime.x);
      float t = tmod(tdelay + S2T * floor(9.0 * dicetime.y), l);
      float q = l - t;
      float st = round((tdelay.w - t) / S2T);

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      env *= exp2(-50.0 * t);

      float progtrans = mod(st, 128.0) < 64.0 ? 0.0 : -3.0;
      int iNote = int(dice.x * float(N_CHORD) * 2.5);
      float pitch = 48.0 + TRANSPOSE + progtrans + float(CHORD[iNote % N_CHORD]);
      pitch += 12.0 * float(iNote / N_CHORD);
      float freq = p2f(pitch);
      float phase = t * freq + dice.y;

      float fm = dice.y * sin(TAU * phase);
      vec2 wave = vec2(sin(TAU * phase + fm));

      sum += delaydecay * env * wave * rotate2D(TAU * float(dice.z));
    }

    dest += 0.12 * sum * mix(0.2, 1.0, duck);
  }

  return clip(1.3 * tanh(dest));
}
