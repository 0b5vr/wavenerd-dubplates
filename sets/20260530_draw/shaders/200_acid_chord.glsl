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
#define TRANSPOSE 6.0

const float SWING = 0.52;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);


uniform vec4 param_knob3; // kick cut

#define p3 paramFetch(param_knob3)

int imod(int x, int y) {
  return ((x % y) + y) % y;
}

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

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

float glidephase(float t, float t1, float p0, float p1) {
  if (p0 == p1) {
    return t * p2f(p1);
  }

  float m0 = (p0 - 69.0) / 12.0;
  float m1 = (p1 - 69.0) / 12.0;
  float b = (m1 - m0) / t1;

  return (
    + p2f(p0) * (pow(2.0, b * min(t, t1)) - 1.0) / b / LN2
    + max(0.0, t - t1) * p2f(p1)
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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float duck = smoothstep(0.0, 0.4, time.x) * smoothstep(0.0, 0.001, B2T - time.x);
  bool isFillIn = false;
  // { // enable fill-in
  //   isFillIn = time.z > 60.0 * B2T;
  // }

  const int N_CHORD_NOTES = 8;
  const int N_CHORD_PROGS = 4;
  const int CHORDS[] = int[](
    0, 7, 10, 12, 14, 17, 19, 22,
    -3, 4, 7, 12, 14, 16, 19, 23,
    -4, 3, 7, 10, 12, 15, 19, 24,
    1, 8, 10, 12, 15, 17, 19, 24
  );

  // { // kick
  //   vec4 seq = seq16(time.y, 0x8888);
  //   if (isFillIn) {
  //     seq = seq16(time.y, 0x8000);
  //   }

  //   float t = seq.t;
  //   float q = seq.q;
  //   duck = min(
  //     duck,
  //     smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
  //   );

  //   {
  //     float env = smoothstep(0.0, 0.001, q);
  //     env *= smoothstep(0.25, 0.1, t);
  //     env *= mix(1.0, exp2(-80.0 * t), p3); // hpf-like

  //     float phase = (
  //       45.0 * t
  //       - 3.0 * exp2(-t * 40.0)
  //       - 5.0 * exp2(-t * 80.0)
  //       - 3.0 * exp2(-t * 400.0)
  //     );

  //     float wave = sin(1.7 * sin(TAU * phase));

  //     dest += 0.6 * env * wave;
  //   }
  // }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-exp2(7.0 - 3.0 * fract(seq.s * 0.41 + 0.6)) * t);

  //   vec2 wave = shotgun(4700.0 * t, 1.8, 0.0, 0.0);
  //   wave = tanh(1.5 * wave);

  //   dest += 0.3 * mix(0.1, 1.0, duck) * env * wave;
  // }

  // { // open hihat
  //   vec4 seq = seq16(time.y, 0x2222);
  //   float t = seq.y;

  //   vec2 sum = vec2(0.0);

  //   float env = exp2(-24.0 * t);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * t + dice2.xy);
  //     wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.15 * mix(0.0, 1.0, duck) * env * tanh(2.0 * sum);
  // }

  // { // ride
  //   vec4 seq = seq16(time.y, 0x2222);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = exp2(-4.0 * t) * smoothstep(0.0, 0.01, q);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 2.9 * env * sin(wave + exp2(13.10 + 0.4 * dice.x) * t + dice2.xy);
  //     wave = 2.8 * env * sin(wave + exp2(14.97 + 0.4 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * env * sin(wave + exp2(14.09 + 1.0 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.06 * env * mix(0.3, 1.0, duck) * tanh(sum);
  // }

  // { // perc
  //   vec4 seq = seq16(time.y, 0x0808);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  //   env *= mix(
  //     exp2(-20.0 * t),
  //     exp2(-4.0 * t),
  //     0.7
  //   );

  //   vec2 wave = mix(
  //     shotgun(1400.0 * t, 1.5, 0.3, 0.0),
  //     cheapnoise(256.0 * t) - cheapnoise(256.0 * t - 0.004),
  //     0.04
  //   );

  //   dest += 0.2 * env * tanh(4.0 * wave);
  // }

  // { // clav
  //   vec4 seq = seq16(time.y, 0xcd52);
  //   float t = seq.y;

  //   float env = mix(
  //     exp2(-100.0 * t),
  //     exp2(-1.0 * t),
  //     0.004
  //   );

  //   vec2 wave = tri(5000.0 * t + vec2(0.0, 0.25));

  //   dest += 0.14 * mix(0.1, 1.0, duck) * env * vec2(wave);
  // }

  // { // crash
  //   float t = mod(time.z, 64.0 * B2T);
  //   if (isFillIn) {
  //     t = time.y;
  //   }

  //   float env = mix(exp(-t), exp(-10.0 * t), 0.7);
  //   vec2 wave = shotgun(4500.0 * t, 1.4, 0.0, 1.0);
  //   dest += 0.4 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  // }

  // { // acid
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float s = seq.s;
  //   float t = seq.t;
  //   float q = seq.q;
  //   q -= S2T * exp2(-1.0 - 4.0 * fract(0.421 * s));

  //   float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);

  //   float cutoff = (
  //     6.0
  //     + 2.0 * smoothstep(0.0, 0.01, t) * exp(-20.0 * t)
  //     + 3.0 * fract(0.421 * seq.s)
  //   );
  //   float cfreq = exp2(cutoff);
  //   float reso = 0.8;

  //   int i = 0;
  //   float pitch = 24.0 + TRANSPOSE;
  //   pitch += 12.0 * floor(2.7 * fract(0.311 * seq.s + 0.4));
  //   float basefreq = p2f(pitch);
  //   float basephase = t * basefreq;

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 128) { // acid
  //     float fi = float(i);
  //     vec3 dice = hash3f(vec3(i) + vec3(1, 5, 7));

  //     float p = 1.0 + 1.0 * fi;
  //     float freq = basefreq * p;
  //     float coeff = exp(-0.1 * p);

  //     vec2 filt = ladderLPF(freq, cfreq, reso);
  //     float phase = basephase * p;
  //     // phase += TAU * dice.z;

  //     vec2 wave = vec2(0.0);
  //     wave += sin(TAU * phase + filt.y);
  //     sum += wave * env * coeff * filt.x;
  //   }

  //   float bias = -0.4;
  //   sum = clip(4.0 * (sum + bias)) - bias;

  //   { // sub
  //     float pitch = 24.0 + TRANSPOSE;
  //     float freq = p2f(pitch);
  //     float phase = t * freq;

  //     float wave = mix(
  //       sin(TAU * phase),
  //       sin(TAU * 2.0 * phase),
  //       0.4
  //     );
  //     wave = tanh(1.5 * wave);

  //     sum += env * wave;
  //   }

  //   dest += 0.3 * mix(0.2, 1.0, duck) * sum;
  // }

  { // chord
    const int N_CHORD_NOTES = 5;
    const int CHORD[] = int[](0, 3, 7, 10, 14);

    vec2 sum = vec2(0);
    repeat(i, 64) {
      vec3 dice = hash3f(vec3(i, 11, 33));

      vec4 timeDelayed = mod(time - 2.0 * S2T * dice.z, timeLength);
      float s = 0.0;
      float l = 16.0 * B2T;
      float t = tmod(timeDelayed, l);
      float pitchOff = 0.0;

      #define S(x) s2tSwing(x)
      #define SEQ(a, b, p) if(t > S(a)) { s = S(a); l = S(b) - s; pitchOff = float(p); }
      SEQ(0, 32, 0)
      SEQ(32, 48, 7)
      SEQ(48, 64, 3)

      t = t - s;

      float env = smoothstep(0.0, S2T, t) * smoothstep(0.0, S2T, l - t);

      vec2 pitch = 48.0 + TRANSPOSE + float(CHORD[i % N_CHORD_NOTES]) + 0.2 * boxMuller(dice.xy);
      pitch += pitchOff;

      vec2 freq = p2f(pitch);
      vec2 phase = t * freq + hash3f(dice).xy;

      float off = 0.2 * (1.0 - exp2(-t - 0.1));
      vec2 wave = cheapfiltersaw(phase, 0.99) + cheapfiltersaw(phase - off, 0.99);

      sum += env * wave * rotate2D(float(i));
    }
    dest += 1.0 * mix(0.2, 1.0, duck) * sum / 64.0;
  }

  return clip(1.3 * tanh(dest));
}
