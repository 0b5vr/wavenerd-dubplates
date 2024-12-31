#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(i) clamp(i, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

const float SWING = 0.51;

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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float duck = 1.0;

  const int N_CHORD = 8;
  int CHORD[] = int[](
    7, 10, 12, 14, 15, 17, 19, 22
  );

  { // kick
    vec4 seq = seq16(time.y, 0x9192);
    float t = seq.t;
    float q = seq.q;
    duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * exp2(-20.0 * max(t - 0.1, 0.0));

    // {
    //   env *= exp2(-40.0 * t);
    // }

    {
      float envd = mix(1.0, 0.7, step(6.0, mod(seq.s, 8.0)));
      float imp = exp2(-t * 800.0);

      float phase = (
        42.0 * t
        - 6.0 * envd * exp2(-t * 20.0)
        - 6.0 * envd * exp2(-t * 50.0)
        - 2.0 * imp
      );
      float wave = sin(TAU * phase + 0.3 * cos(TAU * phase));
      wave -= imp;
      dest += 0.6 * clip(tanh(2.0 * env * wave));
    }
  }

  { // sub kick
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
    env *= exp2(-8.0 * t);

    {
      float wave = sin(TAU * (
        46.0 * t
      ));
      dest += 0.5 * duck * tanh(2.0 * env * wave);
    }
  }

  { // hihat
    float st;
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float vel = fract(0.417 * seq.s);

    float env = smoothstep(0.0, 0.01, q);
    float envk = exp2(mix(7.0, 5.0, vel));
    env *= exp2(-envk * t);

    vec2 wave = shotgun(7000.0 * t, 1.4, 0.1, 0.4);
    dest += 0.3 * env * mix(0.3, 1.0, duck) * tanh(16.0 * wave);
  }

  { // shaker
    vec4 seq = seq16(time.x, 0xffff);
    float t = seq.t;
    float st = seq.s;

    float vel = fract(st * 0.58 + 0.13);
    float env = smoothstep(0.0, 0.02, t) * exp(-40.0 * exp2(-1.0 * vel) * t);

    vec2 sum = vec2(0.0);
    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.2 * exp2(-20.0 * t) * sin(wave + exp2(9.60 * exp2(0.2 * vel) + 0.4 * dice.x) * t + dice2.xy);
      wave = 8.8 * exp2(-1.0 * t) * sin(wave + exp2(10.28 + 0.5 * dice.y) * t + dice2.yz);
      wave = exp2(-10.0 * t) * sin(wave + exp2(14.12 + 0.4 * dice.z) * t + dice2.zx);

      sum += wave / 4.0;
    }

    dest += 0.2 * env * duck * tanh(2.0 * sum);
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
      wave = 2.9 * exp(-1.0 * t) * sin(wave + exp2(13.70 + 0.4 * dice.x) * t + dice2.xy);
      wave = 2.8 * exp(-1.0 * t) * sin(wave + exp2(14.67 + 0.4 * dice.y) * t + dice2.yz);
      wave = 1.0 * sin(wave + exp2(14.09 + 0.5 * dice.z) * t + dice2.zx);

      sum += wave / 4.0;
    }

    dest += 0.05 * env * mix(0.3, 1.0, duck) * tanh(2.0 * sum);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x2424);
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      mix(
        exp2(-60.0 * t),
        exp2(-5.0 * t),
        0.03
      ),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-100.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(1100.0 * t), 1040.0 * t), 0.5, 2.0).xy;

    dest += 0.18 * tanh(20.0 * env * wave);
  }

  { // rim
    vec4 seq = quant(tmod(time, 5.0 * S2T), 5.0 / 3.0);
    float t = seq.t;

    float env = step(0.0, t) * exp2(-400.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * mix(0.3, 1.0, duck) * vec2(wave) * rotate2D(seq.s);
  }

  { // noise
    float l = 32.0 * S2T;
    float t = time.z;
    float q = l - t;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);
    env *= exp2(-0.2 * t);

    vec2 wave = cheapnoise(128.0 * t) - cheapnoise(128.0 * t + 0.002);
    dest += 0.11 * mix(0.2, 1.0, duck) * env * tanh(3.0 * wave);
  }

  { // beep
    vec4 seq = quant(tmod(time, 6.0 * S2T), 3.0);
    float t = seq.t;
    float q = S2T - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

    float phase = 1500.0 * t;
    phase += 500.0 * exp(-30.0 * t) / 30.0;
    phase += 0.08 * sin(8.0 * phase);

    vec2 wave = cis(TAU * phase);

    dest += 0.06 * mix(0.3, 1.0, duck) * env * tanh(1.0 * wave);
  }

  { // sonar
    float t = mod(time.y - 9.0 * S2T, 4.0 * B2T);
    float q = 4.0 * B2T - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp2(-2.0 * t);

    vec2 sum = vec2(0.0);
    repeat(i, 64) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(i, 29, 11));
      vec2 dicen = boxMuller(dice.xy);

      vec2 phase = vec2(1070.0 * t);
      phase *= exp2(0.06 * dicen.x);
      phase += dice.z;
      // phase += 0.1 * sin(10.0 * phase);

      vec2 wave = sin(TAU * phase) * rotate2D(TAU * (dice.z - 0.5) * (1.0 - exp2(-4.0 * t)));
      sum += wave / 32.0;
    }

    dest += 0.8 * mix(0.3, 1.0, duck) * env * clip(sum);
  }

  { // chord
    int N_CHORD = 4;
    int CHORD[] = int[](0, 4, 6, 11);

    int N_PATTERN = 5;
    int PATTERN[] = int[](0, 6, 3, 14, 10);

    vec2 sum = vec2(0.0);
    repeat(iDelay, 3) {
      float fiDelay = float(iDelay);
      float offset = -2.0 * S2T * fiDelay;

      vec4 seq = quant(time.y + offset, 1.0);
      float t = seq.t;
      float q = seq.q;
      float st = seq.s + 16.0 * floor((time.z - offset) / (4.0 * B2T));
      int ip = int(mod(st, float(N_PATTERN)));

      float env = smoothstep(0.0, 0.004, t) * smoothstep(0.0, 0.004, q);
      env *= exp2(-1.0 * t);

      float fenv = exp2(-8.0 * t);
      float lpfcut = 200.0 * exp2(6.0 * fenv - 0.5 * fiDelay);
      float hpfcut = 700.0 * exp2(0.3 * fiDelay);

      vec2 sumn = vec2(0.0);
      repeat(iNote, 4) {
        float pitch = 60.0 + TRANSPOSE + float(CHORD[iNote]);
        pitch += float(PATTERN[ip]);
        float freq = p2f(pitch);

        repeat(iPart, 32) {
          vec3 dice = hash3f(vec3(iPart, 29, 79));
          float p = float(iPart + 1);
            float amp = exp2(-4.0 * dice.x) / p;

          float freqp = p * freq;
          float phase = TAU * freqp * t;

          vec2 lpf = ladderLPF(freqp, lpfcut, 0.0);
          vec2 hpf = twoPoleHPF(freqp, hpfcut, 0.0);
          amp *= lpf.x * hpf.x;
          phase += lpf.y + hpf.y;

          vec2 wave = cis(phase) * rotate2D(TAU * dice.y + time.z);
          sumn += amp * wave;
        }
      }

      sum += env * sumn * exp2(-0.8 * fiDelay);
    }

    dest += 0.4 * mix(0.3, 1.0, duck) * sum;
  }

  return clip(1.3 * tanh(dest));
}
