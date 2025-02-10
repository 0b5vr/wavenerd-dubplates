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

const float SWING = 0.54;

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

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    // {
    //   env *= exp(-60.0 * t);
    // }

    {
      float wave = sin(TAU * (
        45.0 * t
        - 7.0 * exp2(-t * 40.0)
      ));
      dest += 0.5 * clip(4.0 * env * wave);
    }
  }

  { // sub kick
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    {
      float wave = sin(TAU * (
        42.0 * t
        - 4.0 * exp2(-t * 40.0)
      ));
      dest += 0.3 * duck * tanh(2.0 * env * wave);
    }
  }

  { // hihat
    float st;
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = exp(-exp2(5.0) * t) * smoothstep(0.0, 0.01, q);
    vec2 wave = shotgun(3700.0 * t, 2.4, 0.0, 1.0);
    dest += 0.15 * env * duck * tanh(8.0 * wave);
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.t;
    float q = seq.q;

    vec2 sum = vec2(0.0);

    float env = linearstep(0.0, 0.01, q) * mix(
      exp2(-80.0 * max(t - 0.02, 0.0)),
      exp2(-1.0 * t),
      0.01
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

    dest += 0.2 * mix(0.2, 1.0, duck) * env * tanh(2.0 * sum);
  }

  { // perc
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float s = mod(seq.s, 8.0);

    float kdecay = exp2(mix(7.0, 4.0, fract(0.7 + 0.421 * s)));
    float env = exp2(-kdecay * t);

    float freq = exp2(mix(8.0, 10.0, fract(0.282 + 0.384 * s)));
    float phase = freq * t;

    vec3 p = vec3(5, 1, 8);
    p += vec3(cis(TAU * phase), 6.0 * phase);
    vec2 wave = cyclic(p, 0.5, 1.5).xy;

    dest += 0.14 * mix(0.3, 1.0, duck) * env * tanh(4.0 * wave);
  }

  { // hi fm perc
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;

    vec3 dice = hash3f(vec3(seq.s, mod(floor(time.z / 4.0 / B2T), 4.0), 15));

    float env = exp(-50.0 * t);

    float freq = exp2(10.0);
    float phase = freq * t;

    float fm = 14.0 * exp(-10.0 * t) * sin(2.2 * TAU * phase);
    vec2 wave = vec2(sin(TAU * phase + fm)) * rotate2D(seq.s);

    dest += 0.12 * mix(0.3, 1.0, duck) * env * wave;
  }

  { // clav
    vec4 seq = seq16(time.y, 0x5555);
    float t = seq.t;
    float q = S2T - t;

    float env = linearstep(0.0, 0.001, q) * exp(-100.0 * t);

    vec2 wave = cis(-TAU * 4300.0 * t);

    dest += 0.14 * mix(0.3, 1.0, duck) * env * wave;
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0002);
    float t = seq.t;

    float env = mix(
      mix(
        exp2(-40.0 * t),
        exp2(-500.0 * mod(t, 0.012)),
        exp2(-120.0 * max(0.0, t - 0.02))
      ),
      exp(-t),
      0.002
    );

    vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0).xy;

    dest += 0.18 * tanh(20.0 * env * wave);
  }

  { // ride
    float t = mod(time.x - 0.5 * B2T, 1.0 * B2T);

    float env = smoothstep(0.7, 0.0, t);
    float env2 = exp(-2.0 * t);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      float tt = 1.1 * t;

      vec2 wave = vec2(0.0);
      wave = 2.5 * env2 * sin(wave + exp2(14.90 + 0.1 * dice.x) * tt + dice2.xy);
      wave = 4.2 * env2 * sin(wave + exp2(13.87 + 0.1 * dice.y) * tt + dice2.yz);
      wave = 1.0 * env2 * sin(wave + exp2(13.89 + 2.0 * dice.z) * tt + dice2.zx);

      sum += wave;
    }

    dest += 0.05 * mix(0.3, 1.0, duck) * env * tanh(sum);
  }

  { // additive riff
    vec2 sum = vec2(0.0);
    repeat(iDelay, 4) {
      float fiDelay = float(iDelay);
      float delayoff = -2.0 * S2T * fiDelay;

      vec4 timet = mod(time + delayoff, timeLength);
      vec4 seq = seq16(timet.x, 0xffff);
      float t = seq.t;
      float q = seq.q;
      float st = seq.s + 4.0 * floor(timet.z / B2T);

      vec3 dices = hash3f(vec3(st, 4, 11));
      float gate = smoothstep(0.5, 0.7, dices.x + 0.5 * sin(TAU * (st / 32.0 - 0.1)));
      float env = gate * smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.02, q);

      float stmod = fract(0.421 * st);

      float cutenv = exp2(-24.0 * t);
      float cutoff = exp2(
        7.5
        - 0.5 * fiDelay
        + 2.0 * gate
        + 2.0 * stmod
        + 2.0 * cutenv
      );

      vec2 sump = vec2(0.0);
      repeat(i, 128) {
        float fi = float(i);
        vec3 dicep = hash3f(vec3(fi, 24, 47));

        // partial mod
        float p = 1.0 + fi;
        p = pow(p, 1.2); // metal
        p = mix(p, 1.0, 0.2); // centroid
        p *= exp2(0.1 * dicep.x); // random
        float freq = 140.0 * p;

        vec2 phase = vec2(t * freq);
        vec2 lpf = ladderLPF(freq, cutoff, 0.2);
        vec2 hpf = twoPoleHPF(freq, 400.0, 0.0);

        sump += sin(TAU * phase + lpf.y + hpf.y) / p * lpf.x * hpf.x * env * rotate2D(2.4 * fi);
      }
      sum += exp(-0.5 * fiDelay) * tanh(5.0 * sump);
    }

    dest += 0.22 * mix(0.1, 1.0, duck) * sum;
  }

  { // fm drone
    float t = time.z;

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi, 58, 11));
      vec2 dicen = boxMuller(dice.xy);

      float freq = 70.0 * exp2(0.004 * dicen.x);
      float phase = freq * t + dice.z;

      float fm = exp2(4.0 * sin(PI * exp2(-0.4 * t))) * sin(9.78 * TAU * phase);
      vec2 wave = cis(TAU * phase + fm);

      sum += wave / 8.0;
    }

    dest += 0.1 * mix(0.1, 1.0, duck) * sum;
  }

  return clip(1.3 * tanh(dest));
}
