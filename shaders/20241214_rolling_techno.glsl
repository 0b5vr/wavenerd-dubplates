#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(i) clamp(i, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p-0.25)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

const float SWING = 0.52;

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

vec2 spray(float t, float freq, float spread, float seed, float interval, int count) {
  float grainLength = float(count) * interval;

  vec2 sum = vec2(0.0);
  repeat(i, count) {
    float fi = float(i);

    float off = -interval * fi;
    float tg = mod(t + off, grainLength);
    float prog = tg / grainLength;

    vec3 dice = hash3f(vec3(i, lofi(t + off, grainLength), seed));
    vec2 dicen = boxMuller(dice.xy);

    float envg = smoothstep(0.0, 0.5, prog) * smoothstep(1.0, 0.5, prog);

    vec2 phase = vec2(freq * t);
    phase *= exp2(spread * dicen.xy);
    phase += dice.xy;

    vec2 wave = sin(TAU * phase);
    sum += 2.0 * envg * wave;
  }

  return sum / float(count);
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
    //   env *= exp(-40.0 * t);
    // }

    {
      float wave = sin(TAU * (
        48.0 * t
        - 4.0 * exp2(-t * 30.0)
        - 3.0 * exp2(-t * 90.0)
      ));
      dest += 0.6 * tanh(4.0 * env * wave);
    }
  }

  { // rumble
    float t = time.x;
    float q = B2T - t;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    vec2 wave = cheapnoise(t * (0.8 + 0.8 * t));
    wave = tanh(2.0 * wave);
    wave = mix(wave, vec2(dot(wave, vec2(0.5))), 0.5);
    dest += 0.4 * duck * env * wave;
  }

  { // fm bass
    float t0 = tmod(time, 5.0 * S2T);
    vec4 seq = seq16(t0, 0xffff);
    float s = seq.s;
    float t = seq.t;
    float q = seq.q;

    float env = linearstep(0.0, 0.001, t) * linearstep(0.0, 0.01, q) * exp2(-5.0 * t);

    float fmenv = exp2(-2.0 + fract(0.4 + s * 0.288)) * exp2(-30.0 * t);

    float pitch = 30.0 + 8.0 * sin(3.6 * s) + TRANSPOSE;
    float freq = p2f(pitch);
    vec2 phase = vec2(freq * t);
    phase += fmenv * cis(7.324 * TAU * freq * t + time.z);
    vec2 wave = vec2(sin(TAU * phase));

    dest += 0.5 * duck * env * wave;
  }

  { // hihat
    float st;
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = exp(-50.0 * t) * smoothstep(0.0, 0.01, q);
    vec2 wave = shotgun(3200.0 * t, 1.8, 0.0, 2.0);
    dest += 0.15 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.t;
    float q = seq.q;

    vec2 sum = vec2(0.0);

    float env = linearstep(0.0, 0.01, q) * mix(
      exp2(-40.0 * max(t - 0.02, 0.0)),
      exp2(-1.0 * t),
      0.01
    );

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.2 * exp2(-5.0 * t) * sin(wave + exp2(12.10 + 0.2 * dice.x) * t + dice2.xy);
      wave = 4.8 * exp2(-1.0 * t) * sin(wave + exp2(11.38 + 0.2 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.72 + 0.3 * dice.z) * t + dice2.zx);

      sum += wave / 4.0;
    }

    dest += 0.3 * mix(0.2, 1.0, duck) * env * tanh(2.0 * sum);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 8.0);

    float vel = fract(st * 0.58 + 0.13);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);

    vec2 sum = vec2(0.0);
    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.2 * exp2(-exp2(5.0 - vel) * t) * sin(wave + exp2(10.60 + 0.4 * dice.x) * t + dice2.xy);
      wave = 8.8 * exp2(-1.0 * t) * sin(wave + exp2(10.28 + 0.5 * dice.y) * t + dice2.yz);
      wave = exp2(-10.0 * t) * sin(wave + exp2(12.12 + 0.4 * dice.z) * t + dice2.zx);

      sum += wave / 4.0;
    }

    dest += 0.2 * env * duck * tanh(2.0 * sum);
  }

  { // clap
    vec2 sum = vec2(0.0);
    repeat(iDelay, 7) {
      float fiDelay = float(iDelay);
      float modDelay = exp2(-0.8 * fiDelay);

      vec4 seq = seq16(mod(time - 2.0 * S2T * fiDelay, timeLength).y, 0x0100);
      float t = seq.t;

      float env = mix(
        mix(
          exp2(-exp2(mix(4.5, 5.5, modDelay)) * t),
          exp2(-400.0 * mod(t, 0.014)),
          exp2(-180.0 * max(0.0, t - 0.02))
        ),
        exp(-t),
        0.002
      );

      vec2 wave = cyclic(vec3(4.0 * cis(1700.0 * t), 1140.0 * t), mix(0.9, 0.5, modDelay), 4.0).xy;

      sum += modDelay * tanh(20.0 * env * wave);
    }

    dest += 0.16 * mix(0.7, 1.0, duck) * sum;
  }

  { // ride
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.y;
    float q = seq.w;

    float env = exp2(-4.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 2.9 * exp(-1.0 * t) * sin(wave + exp2(13.10 + 0.4 * dice.x) * t + dice2.xy);
      wave = 2.8 * exp(-1.0 * t) * sin(wave + exp2(14.97 + 0.4 * dice.y) * t + dice2.yz);
      wave = 1.0 * sin(wave + exp2(14.09 + 1.0 * dice.z) * t + dice2.zx);

      sum += wave / 4.0;
    }

    dest += 0.08 * env * mix(0.3, 1.0, duck) * tanh(2.0 * sum);
  }

  { // additive riff
    vec2 sum = vec2(0.0);
    repeat(iDelay, 7) {
      float fiDelay = float(iDelay);
      float modDelay = exp2(-0.2 * fiDelay);
      float delayoff = -2.0 * S2T * fiDelay;

      vec4 timet = mod(time + delayoff, timeLength);
      vec4 seq = seq16(timet.y, 0x4343);
      float t = seq.t;
      float q = seq.q;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp(-2.0 * t);

      float accent = float[](0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2)[int(seq + 8.0) % 8];
      float cutenv = smoothstep(0.0, 0.02, t) * exp2(-exp(3.0 - 2.0 * accent) * t);
      float cutoff = exp2(
        7.5
        - 0.5 * fiDelay
        + 5.0 * cutenv
      );

      vec2 sump = vec2(0.0);
      repeat(i, 128) {
        float fi = float(i);
        vec3 dicep = hash3f(vec3(fi, 5, 18));

        // partial mod
        float p = 1.0 + fi;
        p = pow(p, 1.2); // metal
        p = mix(p, 4.0, 0.2); // centroid
        p *= exp2(0.02 * dicep.x); // random
        float freq = 140.0 * p;

        vec2 phase = vec2(t * freq);
        vec2 lpf = ladderLPF(freq, cutoff, 0.2);
        vec2 hpf = twoPoleHPF(freq, mix(2400.0, 800.0, modDelay), 0.0);

        sump += sin(TAU * phase + lpf.y + hpf.y) / p * lpf.x * hpf.x * env * rotate2D(2.4 * fi + time.z);
      }

      float am = 1.0; // sin(TAU * 400.0 * t);
      sum += modDelay * tanh(5.0 * sump * am);
    }

    dest += 0.32 * mix(0.1, 1.0, duck) * sum;
  }

  return clip(1.3 * tanh(dest));
}
