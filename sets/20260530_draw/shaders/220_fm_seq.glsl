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
#define TRANSPOSE 0.0

const float SWING = 0.5;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

uniform vec4 param_knob0; // pad volume
uniform vec4 param_knob3; // kick cut
uniform vec4 param_knob4; // pad mod

#define p0 paramFetch(param_knob0)
#define p3 paramFetch(param_knob3)
#define p4 paramFetch(param_knob4)

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

const int N_CHORD_NOTES = 8;
const int CHORD[] = int[](0, 7, 10, 12, 14, 15, 17, 19, 22);

vec2 mainAudioDry(vec4 time) {
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

    float env = smoothstep(0.0, 0.001, q);
    env *= smoothstep(0.3, 0.1, t);
    env *= mix(1.0, exp2(-50.0 * t), p3);

    vec2 phase = vec2(
      40.0 * t
      - 1.0 * exp2(-t * 10.0)
      - 1.0 * exp2(-t * 80.0)
      - 2.0 * exp2(-t * 200.0)
    );

    vec2 wave = tanh(2.0 * sin(TAU * phase));

    dest += 0.7 * env * wave;
  }

  { // bass
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float bassduck = smoothstep(0.0, B2T, time.x);
    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
    env *= exp2(-20.0 * t);

    float pitch = 36.0 + TRANSPOSE;
    float freq = p2f(pitch);
    float phase = freq * t;

    vec2 osc = vec2(0.0);
    osc += sin(TAU * phase);

    dest += 0.5 * mix(0.7, 1.0, duck) * bassduck * env * osc;
  }

  { // rim
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s + 16.0 * floor(time.z / (16.0 * S2T));
    float t = seq.t;

    float env = exp(-300.0 * t);

    float gate = step(fract(st * 0.721 + 0.2), 0.6);
    vec2 osc = gate * vec2(tanh(4.0 * (
      +tri(t * 400.0 - 0.5 * env)
      +tri(t * 1500.0 - 0.5 * env)
    )));
    osc *= rotate2D(st);
    dest += 0.14 * mix(0.5, 1.0, duck) * env * osc;
  }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float accent = seq16(time.y, 0x0000).s == seq.s ? 1.0 : 0.0;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-mix(100.0, 5.0, accent) * t);

  //   vec2 wave = shotgun(3000.0 * t, 3.0, 0.0, 2.0);
  //   wave = tanh(1.5 * wave);

  //   dest += 0.4 * mix(0.3, 1.0, duck) * env * wave;
  // }

  // { // ride
  //   vec4 seq = seq16(time.y, 0xaaaa);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp(-10.0 * t);

  //     vec2 wave = shotgun(4200.0 * t, 3.0, 0.2, 1.0);
  //   wave = tanh(4.0 * wave);

  //   dest += 0.3 * mix(0.1, 1.0, duck) * env * wave;
  // }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4000.0 * t, 1.9, 0.0, 1.0);
    dest += 0.3 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // riff
    vec2 sum = vec2(0.0);
    repeat(i, 9) {
      vec4 tdelay = mod(time - float(i) * 3.0 * S2T, timeLength);
      vec4 seq = seq16(tdelay.y, 0xffff);
      float st = seq.s + 16.0 * floor(tdelay.z / (16.0 * S2T));
      float t = seq.t;
      float q = seq.q;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp2(-exp2(2.0 + 5.0 * fract(0.626 * st)) * t);

      float pitch = 24.0 + TRANSPOSE;

      { // octave shift
        vec3 dice = hash3f(vec3(st, 10, 18));
        pitch += 12.0 * step(dice.x, 0.4) * floor(1.0 + 4.0 * dice.y);
      }

      { // chord shift
        vec3 dice = hash3f(vec3(st, 34, 28));
        pitch += step(dice.x, 0.2) * float(CHORD[int(float(N_CHORD_NOTES) * dice.y)]);
      }

      { // unquantized shift
        vec3 dice = hash3f(vec3(st, 40, 559));
        pitch += 12.0 * step(dice.x, 0.2) * exp2(dice.y);
      }

      float freq = p2f(pitch);
      float phase = freq * t;

      { // fm
        vec3 dice = hash3f(vec3(st, 128, 111));
        float fmmul = exp2(step(dice.x, 0.2) * exp2(2.0 * dice.y));
        float fm = sin(fmmul * TAU * phase);
        fm *= exp2(-2.0 + 3.0 * fract(0.418 * st));
        phase += fm;
      }

      vec2 osc = vec2(0.0);
      osc += cis(TAU * phase + TAU * time.z / B2T);

      float delaydecay = exp2(-1.0 * float(i));
      sum += delaydecay * env * osc;
    }

    dest += 0.08 * mix(0.5, 1.0, duck) * sum;
  }

  { // chord
    float t = time.z;

    vec2 sum = vec2(0.0);
    repeat(i, 64) {
      vec3 dice = hash3f(vec3(i, 60, 80));

      float pitch = 48.0 + TRANSPOSE + float(CHORD[i % N_CHORD_NOTES]);
      float freq = p2f(pitch);
      float phase = t * freq * exp2(0.02 * (dice.x - 0.5));
      phase += dice.y;

      vec3 d = exp2(-2.0 + 4.0 * p4) * vec3(4.0, -6.0, 2.0);
      vec2 osc = mix(
        cyclic(fract(phase) * d, 0.5, 2.0),
        cyclic((fract(phase) - 1.0) * d, 0.5, 2.0),
        smoothstep(0.9, 1.0, fract(phase))
      ).xy;
      osc *= rotate2D(2.4 * float(i));

      sum += osc / 64.0;
    }
    dest += 0.8 * p0 * mix(0.5, 1.0, duck) * sum;
  }

  { // fm drone
    float t = tmod(time - 32.0 * B2T, timeLength.z);

    float env = smoothstep(0.0, 1.0, t) * exp2(-0.3 * t);

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi, 13, 41));
      vec2 dicen = boxMuller(dice.xy);

      float pitch = 24.0 + TRANSPOSE;
      float freq = p2f(pitch) * exp2(0.009 * dicen.x);
      float phase = freq * t + dice.z;

      // high freq
      phase += 0.2 * fract(phase);

      // fm
      phase += exp2(3.0 * sin(PI * exp2(-0.2 * t))) * sin(7.28 * TAU * phase);

      vec2 osc = cis(TAU * phase);

      sum += osc / 8.0;
    }

    dest += 0.06 * env * mix(0.5, 1.0, duck) * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  dest *= 1.2;
  return dest;
}
