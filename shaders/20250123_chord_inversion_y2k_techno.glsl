#define TRANSPOSE -2.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(i) clamp(i, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p-0.25)-0.5))
#define b2u(b) (0.5+0.5*b)
#define u2b(u) (-1.0+2.0*u)
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

const float SWING = 0.56;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);
const float MIN3 = pow(2.0, 3.0 / 12.0);
const float P4 = pow(2.0, 5.0 / 12.0);
const float P5 = pow(2.0, 7.0 / 12.0);

uniform vec4 param_knob0; // wavefold on osc

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

  const int N_PROG = 4;
  const int N_CHORD = 6;
  const int CHORDS[] = int[](
    -3, 7, 11, 12, 14, 16,
    -7, 7, 8, 10, 12, 15,
    -5, 5, 7, 10, 12, 14,
    -6, 6, 8, 10, 13, 17
  );

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);
    duck = pow(duck, 2.0);

    float env = smoothstep(0.0, 0.001, q) * exp2(-20.0 * max(t - 0.1, 0.0));

    // {
    //   env *= exp2(-80.0 * t);
    // }

    {
      float phase = (
        42.0 * t
        - 4.0 * exp2(-t * 30.0)
        - 4.0 * exp2(-t * 70.0)
      );
      float wave = sin(TAU * phase + 0.3 * cos(TAU * phase));
      dest += 0.6 * tanh(2.0 * env * wave);
    }
  }

  { // hihat
    const int i_patternCH = 0xffff;
    vec4 seq = seq16(time.y, i_patternCH);
    float t = seq.y;

    float vel = fract(seq.x * 0.38);
    float env = exp2(-exp2(6.0 - 1.0 * vel - float(mod(seq.x, 4.0) == 2.0)) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 1.0);
    dest += 0.16 * mix(0.2, 1.0, duck) * env * tanh(8.0 * wave);
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.y;

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.1 * mix(0.2, 1.0, duck) * exp2(-14.0 * t) * tanh(2.0 * sum);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0808);
    float t = seq.y;
    float q = seq.w;

    float env = mix(
      exp2(-60.0 * t),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-100.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(1200.0 * t), 1840.0 * t), 0.5, 2.0).xy;

    dest += 0.2 * tanh(20.0 * env * wave);
  }

  { // ride
    vec4 seq = seq16(time.y, 0xaaaa);
    float t = seq.y;
    float q = seq.w;

    float env = exp2(-4.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 2.1 * env * sin(wave + exp2(13.80 + 0.1 * dice.x) * t + dice2.xy);
      wave = 2.8 * env * sin(wave + exp2(14.67 + 0.2 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(14.09 + 0.2 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.1 * mix(0.1, 1.0, duck) * env * tanh(sum);
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 3.0);
    dest += 0.4 * env * mix(0.2, 1.0, duck) * tanh(8.0 * wave);
  }

  { // bass
    const int N_NOTES = 8;
    const int NOTES[] = int[](
      0, 0, -5, 12, 12, -7, 7, 7
    );

    vec4 seq = seq16(time.y, 0xb6b6);
    float st = mod(seq.s, 0.4);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q - 0.02);

    float cutoff = 8.0;
    // cutoff += -3.0 + 6.0 * paramFetch(param_knob0);
    cutoff += 3.0 * exp(-8.0 * t);
    float cfreq = exp2(cutoff);
    float reso = 0.8;
    // reso = paramFetch(param_knob1);

    int prog = int(mod(time.z / (4.0 * B2T), 4.0));
    float pitch = 36.0 + TRANSPOSE + float(CHORDS[N_CHORD * prog]) + float(NOTES[int(seq.s) % N_NOTES]);
    float freq = p2f(pitch);

    vec2 sum = vec2(0.0);

    { // sub
      float phase = t * freq;
      float wave = sin(TAU * phase);
      sum += 0.5 * tanh(1.5 * wave);
    }

    { // wavetable
      const float DELTA_PHASE = 1.0 / 64.0;
      float phase = t * freq;
      float phasei0 = lofi(phase, DELTA_PHASE);
      float phasei1 = phasei0 + DELTA_PHASE;
      float phaseif = fract(phase / DELTA_PHASE);

      repeat(i, 2) {
        float fi = float(i);
        float phase = mix(phasei0, phasei1, fi);
        float phaser = 12.0 * exp2(-0.2 * floor(phase)) * fract(2.0 * phase);

        float square = 1.0 - 2.0 * step(0.5, fract(phase));
        float wave = mix(
          square,
          -square * cos(TAU * phaser),
          exp(-0.5 * phaser) * smoothstep(1.0, 0.8, fract(2.0 * phase))
        ) * exp2(-0.02 * phase);
        sum += mix(1.0 - phaseif, phaseif, fi) * wave;
      }
    }

    dest += 0.4 * env * mix(0.04, 1.0, duck) * sum;
  }

  { // lead
    vec2 sum = vec2(0.0);

    repeat(iDelay, 4) {
      float fiDelay = float(iDelay);
      float offset = -2.8 * S2T * fiDelay;

      vec2 sumd = vec2(0.0);

      vec4 seq = seq16(time.y + offset, 0xffff);
      float t = seq.t;
      float q = seq.q;
      float st1 = seq.s + mod(16.0 * floor((time.z + offset) / (4.0 * B2T)), 64.0);
      float st0 = mod(st1 - 1.0, 64.0);

      float env = smoothstep(0.0, 0.002, t) * smoothstep(0.0, 0.002, q - 0.01);

      int prog0 = int(mod(floor(st0 / 16.0), 4.0));
      int prog1 = int(mod(floor(st1 / 16.0), 4.0));
      int inv0 = int(11.0 * fract(0.5 + 0.408 * st0));
      int inv1 = int(11.0 * fract(0.5 + 0.408 * st1));

      const int N_UNISON = 3;

      repeat(iNote, 5) {
        int iinvn = 6 + int[](0, 1, 2, 3, 4)[iNote];
        int iinv0 = iinvn + inv0;
        int iinv1 = iinvn + inv1;
        int i0 = (iinv0 % 5) + 1 + N_CHORD * prog0;
        int i1 = (iinv1 % 5) + 1 + N_CHORD * prog1;
        float pitch0 = 36.0 + TRANSPOSE + float(CHORDS[i0]) + 12.0 * float(iinv0 / 5);
        float pitch1 = 36.0 + TRANSPOSE + float(CHORDS[i1]) + 12.0 * float(iinv1 / 5);

        repeat(iUnison, N_UNISON) {
          vec3 dice = hash3f(vec3(iNote, iUnison, 78));

          vec2 phase = vec2(glidephase(t, 0.025, pitch0, pitch1));
          phase *= exp2(0.002 * u2b(float(iUnison) / float(N_UNISON - 1)));
          phase += dice.xy;

          float k = 1.0 - exp2(-6.0 + mix(3.0, 0.0, exp2(-fiDelay)));
          vec2 wave = cheapfiltersaw(phase, k);
          sumd += wave / float(5 * N_UNISON) * rotate2D(TAU * dice.z);
        }
      }

      sum += env * sumd * exp2(-1.2 * fiDelay);
    }

    dest += 0.7 * mix(0.3, 1.0, duck) * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  return clip(1.3 * tanh(dest));
}
