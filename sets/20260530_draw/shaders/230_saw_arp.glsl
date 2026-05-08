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
#define TRANSPOSE -2.0

const float SWING = 0.5;

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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float duck = 1.0;

  #define GET_IPROG(t) step(16.0 * B2T, mod(t, 32.0 * B2T))

  { // kicks
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
      env *= mix(1.0, exp2(-50.0 * t), p3);

      vec2 phase = vec2(
        44.0 * t
        - 1.0 * exp2(-t * 20.0)
        - 1.0 * exp2(-t * 80.0)
        - 2.0 * exp2(-t * 200.0)
      );

      vec2 wave = tanh(2.0 * sin(TAU * phase));

      dest += 0.7 * env * wave;
    }
  }

  { // bass
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
    env *= exp2(-20.0 * t);

    float pitch = 36.0 + TRANSPOSE;
    float freq = p2f(pitch);
    float phase = freq * t;

    vec2 osc = vec2(0.0);
    osc += sin(TAU * phase);

    dest += 0.5 * mix(0.0, 1.0, duck) * env * osc;
  }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-30.0 * t);

  //   vec2 wave = shotgun(4000.0 * t, 2.0, 0.0, 0.0);
  //   wave = tanh(1.5 * wave);

  //   dest += 0.3 * mix(0.3, 1.0, duck) * env * wave;
  // }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4000.0 * t, 1.9, 0.0, 1.0);
    dest += 0.3 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // arp + chord
    const int N_NOTES_ARP = 6;
    const int N_NOTES_CHORD = 8;
    const int NOTES[] = int[](
      // 0, 7, 14, 15, 22, 26, 17, 19 // 9th
      0, 7, 14, 19, 26, 31, 17, 22 // 5th
      // 0, 7, 12, 19, 24, 36, 14, 17 // oct
      // 0, 7, 14, 21, 28, 35, 12, 19 // quintal
    );

    { // arp
      vec2 sum = vec2(0.0);
      repeat(iDelay, 4) {
        vec4 tdelay = mod(time - 3.0 * S2T * float(iDelay), timeLength);

        vec4 seq = seq16(tdelay.y, 0xffff);
        float s = seq.s + 16.0 * floor(tdelay.z / (16.0 * S2T));
        float t = seq.t;
        float q = seq.q;

        vec3 dice = hash3f(vec3(s, 20, 20));

        float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);

        float pitch = 60.0 + TRANSPOSE + float(NOTES[imod(int(s), N_NOTES_ARP)]);
        float freq = p2f(pitch);
        vec2 phase = t * freq * exp2(0.01 * (dice.xy - 0.5));
        phase += dice.yx;

        vec2 osc = vec2(0.0);
        osc += cheapfiltersaw(phase, 1.0);

        float delaydecay = exp2(-1.0 * float(iDelay));
        sum += delaydecay * env * osc;
      }
      dest += 0.07 * mix(0.7, 1.0, duck) * sum;
    }

    // { // chord
    //   float t = time.z;

    //   vec2 sum = vec2(0.0);
    //   repeat(i, 64) {
    //     vec3 dice = hash3f(vec3(i, 60, 80));

    //     float pitch = 48.0 + TRANSPOSE + float(NOTES[i % 8]);
    //     float freq = p2f(pitch);
    //     vec2 phase = t * freq * exp2(0.02 * (dice.xy - 0.5));
    //     phase += dice.yx;

    //     vec2 osc = vec2(0.0);
    //     osc += cheapfiltersaw(phase, 1.0);

    //     sum += osc / 64.0;
    //   }
    //   dest += 0.3 * mix(0.7, 1.0, duck) * sum;
    // }
  }

  return clip(1.3 * tanh(dest));
}
