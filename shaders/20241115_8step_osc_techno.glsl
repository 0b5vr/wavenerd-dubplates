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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float sidechain = 1.0;

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    sidechain = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(2.0 * B2T, 0.1 * B2T, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    {
      float wave = sin(TAU * (
        45.0 * t
        - 7.0 * exp2(-t * 30.0)
        - 2.0 * exp2(-t * 90.0)
        - 4.0 * exp2(-t * 500.0)
        - 0.04 * sin(120.0 * t)
      ));
      dest += 0.5 * tanh(2.0 * env * wave);
    }
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;

    float vel = fract(seq.s * 0.611);
    float env = exp2(-exp2(6.0 - 1.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.0);
    dest += 0.2 * env * mix(0.2, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.t;
    float q = seq.q;

    vec2 sum = vec2(0.0);

    float env = linearstep(0.0, 0.01, q) * mix(
      exp2(-40.0 * t),
      exp2(-1.0 * t),
      0.02
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

    dest += 0.2 * env * mix(0.2, 1.0, sidechain) * tanh(2.0 * sum);
  }

  { // shaker
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;

    float vel = fract(0.62 * seq.s);
    float env = smoothstep(0.0, 0.02, t) * exp2(-exp2(6.0 - vel) * t);
    float pitch = exp2(11.4 + 0.3 * vel);
    vec2 wave = shotgun(pitch * t, 1.0, 0.2, 2.0);
    dest += 0.3 * env * mix(0.2, 1.0, sidechain) * tanh(8.0 * wave);
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
      wave = 2.9 * env * sin(wave + exp2(14.27 + 0.2 * dice.x) * t + dice2.xy);
      wave = 2.2 * env * sin(wave + exp2(13.87 + 0.1 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(13.39 + 0.9 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.04 * env * mix(0.3, 1.0, sidechain) * tanh(sum);
  }

  { // perc
    vec4 seq = seq16(time.y, 0x2020);
    float t = seq.t;

    float env = mix(
      exp2(-40.0 * t),
      exp2(-4.0 * t),
      0.01
    );

    vec2 sum = vec2(0.0);
    repeat(i, 4) {
      vec3 dice = hash3f(vec3(i, 48, 29));

      float freq = 1220.0 * exp2(0.02 * dice.x);
      vec2 phase = dice.yz;
      phase += t * freq;
      phase -= 0.04 * fract(17.0 * phase);
      phase += exp2(-t) * sin(TAU * 1.582 * phase);

      vec2 wave = sin(TAU * phase);
      wave = tanh(2.0 * env * wave);
      sum += wave / 4.0;
    }

    dest += 0.2 * sidechain * sum;
  }

  { // modshit
    vec2 sum = vec2(0.0);

    int STEPS = 8;
    float seqp[] = float[](0.47, 0.73, 0.21, 1.00, 0.02, 0.86, 0.89, 0.23);
    float seqm[] = float[](0.51, 0.52, 0.12, 0.15, 0.31, 0.72, 0.93, 0.48);
    float seqe[] = float[](0.40, 0.70, 0.20, 0.90, 0.10, 0.50, 0.80, 0.20);

    repeat(i, 3) {
      float fi = float(i);
      float delayoff = 2.0 * S2T * fi;
      float delaydecay = exp2(-2.0 * fi);

      vec4 seq = seq16(tmod(time - delayoff, float(STEPS) * S2T), 0xffff);
      float t = seq.t;
      float q = seq.q;

      int si = int(seq.s) % STEPS;
      int sip = (si + STEPS - 1) % STEPS;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp2(-exp2(7.0 - 5.0 * seqe[si]) * t);

      float phase;
      {
        float p0 = 40.0 + 38.0 * seqp[sip];
        float p1 = 40.0 + 38.0 * seqp[si];
        phase = glidephase(t, 0.01, p0, p1);
      }
      {
        float p0 = 50.0 + 17.0 * seqm[sip];
        float p1 = 50.0 + 17.0 * seqm[si];
        phase += 0.1 * env * sin(TAU * glidephase(t, 0.01, p0, p1));
      }
      vec2 wave = cis(TAU * phase);

      sum += delaydecay * env * tanh(4.0 * wave);
    }

    dest += 0.2 * mix(0.2, 1.0, sidechain) * sum;
  }

  { // oidos drone
    vec2 sum=vec2(0.0);

    repeat(i, 2500) {
      vec3 diceA = hash3f(vec3(i / 50) + vec3(38, 68, 36));
      vec3 diceB = hash3f(vec3(i));

      float t = mod(time.z - diceA.x * (64.0 * B2T), 64.0 * B2T);
      float env = sin(PI * t / (64.0 * B2T));

      float tone = 8.0 + 8.0 * diceA.y + 0.06 * diceB.y;
      float freq = exp2(tone);
      vec2 phase = t * freq + fract(diceB.xy * 999.0);
      phase += 0.1 * fract(32.0 * phase); // add high freq

      sum += sin(TAU * phase) * env / 2000.0;
    }

    dest += 1.0 * mix(0.2, 1.0, sidechain) * sum;
  }

  return clip(1.3 * tanh(dest));
}
