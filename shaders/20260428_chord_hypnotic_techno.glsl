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

const float SWING = 0.54;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

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
    vec4 seqa = seq16(time.y, 0x8888);
    vec4 seqb = seq16(time.y, 0x1212);

    duck = min(
      duck,
      smoothstep(0.0, 0.4, seqa.t) * smoothstep(0.0, 0.001, seqa.q)
    );

    { // main kick
      float t = min(seqa.t, seqb.t);
      float q = min(seqa.q, seqb.q);

      if (seqa.t == t) {
        float env = smoothstep(0.0, 0.001, q);
        env *= smoothstep(0.3, 0.1, t);

        // { // highpass-like
        //   env *= exp(-50.0 * t);
        // }

        vec2 phase = vec2(
          44.0 * t
          - 3.0 * exp2(-t * 40.0)
          - 2.0 * exp2(-t * 60.0)
          - 2.0 * exp2(-t * 100.0)
        );

        vec2 wave = tanh(2.0 * sin(TAU * phase));

        dest += 0.7 * env * wave;
      }
    }

    { // sub kick
      float t = min(seqa.t, seqb.t);
      float q = min(seqa.q, seqb.q);

      if (seqb.t == t) {
        float env = smoothstep(0.0, 0.001, q);
        env *= smoothstep(0.3, 0.2, t);

        // { // highpass-like
        //   env *= exp(-50.0 * t);
        // }

        float phase = (
          48.0 * t
          - 2.0 * exp2(-t * 60.0)
        );

        float wave = sin(TAU * phase);

        dest += 0.5 * duck * env * wave;
      }
    }
  }

  { // bass
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    float freq = 50.0;
    float phase = freq * t;
    vec2 wave = vec2(tanh(sin(TAU * phase)));

    dest += 0.6 * mix(0.0, 1.0, duck) * env * wave;
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp(-50.0 * t);

    vec2 wave = shotgun(5400.0 * t, 1.5, 0.0, 0.0);
    wave = tanh(1.5 * wave);

    dest += 0.3 * mix(0.1, 1.0, duck) * env * wave;
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.y;

    vec2 sum = vec2(0.0);

    float env = exp2(-40.0 * t);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.80 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(10.78 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.15 * mix(0.0, 1.0, duck) * env * tanh(2.0 * sum);
  }

  { // ride
    vec4 seq = seq16(time.y, 0x2222);
    float st = seq.s + 16.0 * floor(time.z / 16.0 / S2T);
    float t = seq.y;
    float q = seq.w;

    float lfo = cos(TAU / 256.0 * st);

    float env = exp2(-1.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 2.9 * env * sin(wave + exp2(14.50 + 0.3 * lfo + 0.3 * dice.x) * t + dice2.xy);
      wave = 2.8 * env * sin(wave + exp2(11.97 + 0.3 * lfo + 0.2 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(13.09 + 0.3 * lfo + 0.5 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.02 * env * mix(0.2, 1.0, duck) * tanh(sum);
  }

  { // jet
    float t = tmod(time - s2tSwing(2.0), 8.0 * B2T);

    float env = mix(
      exp2(-8.0 * t),
      exp2(-1.0 * t),
      0.8
    );

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(i / 2, 11, 12));
      float pol = u2b(float(i % 2));
      vec2 dicen = boxMuller(dice.xy);

      float tt = t + 0.003 * fi;

      vec3 p = 140.0 * vec3(3.0, 4.0, 6.0) * tt;
      p.z += 5.0 * tt;
      vec2 wave = cyclic(p, 1.0, 2.0).xy - cyclic(p + 0.1, 1.0, 2.0).xy;
      sum += vec2(wave) * rotate2D(0.7 * fi);
    }

    dest += 0.03 * mix(0.2, 1.0, duck) * env * sum;
  }

  { // fm shot
    float t = tmod(time - s2tSwing(16.0 + 10.0), 32.0 * B2T);

    float env = smoothstep(0.0, 0.001, t) * exp(-4.0 * t);

    vec2 sum = vec2(0.0);
    repeat(i, 32) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(i / 2, 11, 12));
      float pol = u2b(float(i % 2));
      vec2 dicen = boxMuller(dice.xy);

      float freq = 110.0;
      freq *= exp2(0.03 * dicen.x);
      float phase = freq * (t + 0.0003 * pol) + dice.z;
      phase += 0.5 * exp(-0.5 * t) * sin(TAU * 0.62 * phase + sin(TAU * 1.28 * phase));

      float wave = pol * sin(TAU * phase);
      sum += vec2(wave) * rotate2D(TAU * dice.y);
    }

    dest += 0.1 * mix(0.2, 1.0, duck) * env * sum;
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4100.0 * t, 1.9, 0.0, 1.0);
    dest += 0.3 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // riff
    const int N_NOTES = 6;
    const int NOTES[] = int[](0, 6, 12, 0, 5, 8);

    const int N_CHORD = 3;
    const int CHORD[N_CHORD] = int[](0, 6, 7);

    float repeatTime = 9.0 * S2T;

    vec2 sum = vec2(0.0);

    repeat(i, 4) {
      float fi = float(i);

      vec4 tdelay = mod(time - 3.0 * S2T * fi, timeLength);
      vec4 seq = seq16(tdelay.y, 0xffff);
      float st = seq.s + 16.0 * floor(tdelay.z / 16.0 / S2T);
      float t = seq.t;
      float q = seq.q;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp(-10.0 * t);

      float cutoff = exp2(9.0 + 3.0 * env - 0.5 * fi);

      vec2 wave = vec2(0.0);
      repeat(j, N_CHORD) {
        float fj = float(j);

        float note = 48.0 + TRANSPOSE + float(CHORD[j]) + float(NOTES[int(st) % N_NOTES]);
        float freq = p2f(note);

        repeat(k, 128) {
          float fk = float(k);
          float p = 1.0 + fk;
          float freqp = freq * p;

          vec2 lpf = ladderLPF(freqp, cutoff, 0.1);
          vec2 hpf = twoPoleHPF(freqp, 100.0, 0.0);
          float phase = TAU * freqp * t + lpf.y + hpf.y;

          wave += vec2(sin(phase)) / p * lpf.x * hpf.x * rotate2D(2.4 * fk + fi);
        }
      }

      float delaydecay = exp(-1.0 * fi);
      sum += env * delaydecay * tanh(0.5 * wave);
    }

    dest += 0.4 * mix(0.1, 1.0, duck) * sum;
  }

  { // oidos drone
    vec2 sum = vec2(0.0);

    repeat(i, 2500) {
      vec3 dicea = hash3f(vec3(i / 50, 45, 12));
      vec3 diceb = hash3f(vec3(i, 35, 12));
      vec2 dicen = boxMuller(diceb.xy);

      float t = mod(time.z - dicea.x * (64.0 * B2T), 64.0 * B2T);
      float env = sin(PI * t / (64.0 * B2T));

      float tone = 7.0 + 7.0 * dicea.y + 0.02 * dicen.x;
      float freq = exp2(tone);
      vec2 phase = t * freq + fract(diceb.xy * 999.0);

      float amp = 1.0;

      float pm = 0.2;
      phase += pm * fract(2.0 * phase); // add high freq

      sum += amp * env * sin(TAU * phase) / 1000.0;
    }

    dest += 0.3 * mix(0.1, 1.0, duck) * sum;
  }

  return clip(1.3 * tanh(dest));
}
