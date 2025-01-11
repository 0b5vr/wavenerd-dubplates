#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract((p)+0.25)-0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float SWING = 0.52;

const float LN2 = log(2.0);
const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float P5 = pow(2.0, 7.0 / 12.0);

uniform vec4 param_knob0; // pluck envelope

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

float p2f(float p) {
  return exp2((p - 69.0) / 12.0) * 440.0;
}

float cheapFilterSaw(float phase, float k) {
  float wave = fract(phase);
  float c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
}

vec2 cheapFilterSaw(vec2 phase, float k) {
  vec2 wave = fract(phase);
  vec2 c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
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

  const int N_CHORD = 8;
  int CHORDS[N_CHORD * 2] = int[](
    -11, 1, 8, 10, 15, 17, 24, 31,
    -17, -5, 2, 5, 9, 12, 14, 24
  );

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    float env = smoothstep(0.0, 0.001, q) * exp(-20.0 * max(t - 0.12, 0.0));

    // {
    //   env *= exp(-50.0 * t);
    // }

    float tt = t;
    float wave = 0.0;

    wave += tanh(1.5 * sin(TAU * (
      48.0 * tt
      - 4.0 * exp2(-tt * 60.0)
    )));

    float tick = env * exp2(-500.0 * t);
    wave += env * tanh(1.5 * sin(TAU * 3.0 * tick));

    dest += 0.6 * env * wave;
  }

  { // hihat
    float st;
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = exp2(-exp2(6.0) * t) * smoothstep(0.0, 0.01, q);
    vec2 wave = shotgun(3700.0 * t, 2.4, 0.0, 1.0);
    dest += 0.13 * duck * env * tanh(8.0 * wave);
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.t;
    float q = seq.q;

    vec2 sum = vec2(0.0);

    float env = linearstep(0.0, 0.01, q) * mix(
      exp2(-40.0 * t),
      exp2(-1.0 * t),
      0.04
    );

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-3.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.78 + 0.1 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.3 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.1 * mix(0.2, 1.0, duck) * env * tanh(2.0 * sum);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0001);
    float t = seq.t;

    float env = mix(
      exp2(-70.0 * t),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-120.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0).xy;

    dest += 0.2 * tanh(20.0 * env * wave);
  }

  { // click
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;

    vec3 dice = hash3f(vec3(seq.s, mod(floor(time.z / 4.0 / B2T), 4.0), 15));

    float kdecay = exp2(mix(11.0, 5.0, fract(0.421 * seq.s)));
    float env = exp2(-kdecay * t);

    float freq = exp2(mix(8.0, 10.0, dice.y));
    float phase = freq * t;

    vec3 p = vec3(5, 8, 8);
    p += vec3(cis(TAU * phase), 6.0 * phase);
    vec2 wave = cyclic(p, 0.5, 1.2).xy;

    dest += 0.22 * mix(0.3, 1.0, duck) * env * tanh(4.0 * wave);
  }

  { // hi click
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.t;

    vec3 dice = hash3f(vec3(seq.s, mod(floor(time.z / 4.0 / B2T), 4.0), 15));

    t -= lofi(min(t, 0.5 * B2T), 0.0625 * B2T) * step(14.0, seq.s);

    float env = exp(-200.0 * t);

    float freq = exp2(12.0 + lofi(2.0 * dice.y, 0.4));
    float phase = freq * t;

    vec2 wave = vec2(sin(TAU * phase)) * rotate2D(seq.s);

    dest += 0.12 * mix(0.3, 1.0, duck) * env * wave;
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4200.0 * t, 2.0, 0.0, 0.0);
    dest += 0.3 * mix(0.2, 1.0, duck) * env * tanh(8.0 * wave);
  }

  { // bass
    float t = mod(time.z, 8.0 * B2T);
    float q = 8.0 * B2T - t;
    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    int prog = int(time.z / (8.0 * B2T)) & 1;
    float pitch = 48.0 + TRANSPOSE + float(CHORDS[N_CHORD * prog]);
    float freq = p2f(pitch);
    float phase = freq * t;

    float wave = tanh(2.0 * sin(TAU * phase));
    dest += 0.4 * mix(0.0, 1.0, duck) * env * vec2(wave);
  }

  { // chord
    vec2 sum = vec2(0.0);

    repeat(i, 128) {
      int progress = i / 64;

      float l = 8.0 * B2T;
      float t = mod(time.z + (8.0 * float(progress) + 0.5) * B2T, 16.0 * B2T);
      float fade = smoothstep(0.0, B2T, t) * smoothstep(-B2T, 0.0, l - t);

      vec3 dice = hash3f(vec3(i, 7, 7));
      vec2 dicen = boxMuller(dice.xy);

      int iNote = i % N_CHORD;
      float pitch = 48.0 + TRANSPOSE + float(CHORDS[iNote + N_CHORD * progress]);
      float freq = p2f(pitch);

      float phase = t * freq + dice.x;
      phase *= exp(0.01 * dicen.x);
      phase += 0.0001 * freq * sin(30.0 * t + dice.x);
      phase = lofi(phase, 1.0 / 16.0);

      float pers = 0.2;
      pers *= 0.01 + fade;

      vec3 c = vec3(7.0);
      vec3 d = vec3(7.0, -3.0, 2.0);

      vec2 wave = mix(
        vec2(tri(phase)),
        cyclic(fract(phase) * d, pers, 2.0).xy,
        0.4
      );

      float amp = fade * smoothstep(0.0, freq / p2f(36.0 + TRANSPOSE), t / S2T);
      sum += amp * wave * rotate2D(float(i));
    }

    dest += 0.04 * mix(0.2, 1.0, duck) * sum;
  }

  { // pluck chord
    vec2 sum = vec2(0.0);

    repeat(i, 128) {
      float fi = float(i);
      float iDelay = mod(fi, 5.0);

      float t0 = mod(time.z - 0.5 * B2T * iDelay, timeLength.z);
      vec4 seq = quant(t0, 3.0);
      float t = seq.t;
      float q = seq.q;

      vec3 dice = hash3f(vec3(i, 7, 7));
      vec2 dicen = boxMuller(dice.xy);

      float env = exp2(-10.0 * t) * smoothstep(0.0, 0.01, q);

      int prog = int(seq.s / 32.0) & 1;
      int iNote = 2 + i % (N_CHORD - 2);
      float pitch = 48.0 + TRANSPOSE + float(CHORDS[iNote + N_CHORD * prog]);
      float freq = p2f(pitch);

      vec2 phase = (t * freq) * exp(0.001 * dicen.xy);
      phase += dice.xy;

      float kfenv = exp2(4.0 * (1.0 - paramFetch(param_knob0)));
      float fenv = mix(0.5, 1.0, exp2(-kfenv * t)) * exp(-0.1 * iDelay);
      vec2 wave = vec2(cheapFilterSaw(phase, fenv));

      sum += env * wave * rotate2D(float(i)) * exp(-0.5 * iDelay);
    }

    dest += 0.1 * mix(0.2, 1.0, duck) * sum;
  }

  { // fm pluck
    const int N_NOTES = 4;
    const int NOTES[N_NOTES] = int[](7, 0, 2, -5);

    vec2 sum = vec2(0.0);
    repeat(i, 48) {
      float fi = float(i);
      float iDelay = mod(fi, 6.0);
      vec3 dice = hash3f(fi + vec3(14, 25, 30));
      vec2 dicen = boxMuller(dice.xy);

      float t0 = tmod(time - 0.5 * B2T * iDelay, 10.0 / 4.0 * B2T);
      float seqi;
      vec4 seq = quant(t0, 2.5, seqi);
      int inote = clamp(int(seqi), 0, N_NOTES - 1);
      float t = seq.t;
      float q = seq.q;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp(-10.0 * t);

      float pitch = 72.0 + TRANSPOSE + float(NOTES[inote]);
      float freq = p2f(pitch);
      freq *= exp2(0.001 * dicen.x);
      vec2 phase = freq * t + dice.xy;
      phase += 0.05 * cheapnoise(2.0 * t);

      vec2 wave = exp(-1.5 * t) * sin(TAU * phase);
      wave = sin(TAU * phase + wave);

      sum += env * wave * exp(-1.5 * iDelay);
    }

    dest += 0.1 * mix(0.6, 1.0, duck) * sum;
  }

  return clip(1.3 * tanh(dest));
}
