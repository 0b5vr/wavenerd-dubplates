#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define u2b(u) ((u) * 2.0 - 1.0)
#define b2u(b) ((b) * 0.5 + 0.5)
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define TRANSPOSE -1.0

const float SWING = 0.5;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;

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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float duck = 1.0;

  const int N_CODE_NOTES = 8;
  const int CODES[] = int[](
    0, 5, 7, 10, 12, 14, 17, 19,
    -4, 5, 7, 10, 12, 15, 17, 19,
    -7, 3, 7, 10, 12, 14, 17, 19
  );

  // { CHORD PROG }
  #define CHORD_PROG 0

  #define CHORD_NOTE(i) (float(CODES[CHORD_PROG * N_CODE_NOTES + int(i) % N_CODE_NOTES]))

  { // kick
    vec4 seq = seq16(time.y, 0x8890);
    float t = seq.t;
    float q = seq.q;
    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    {
      float env = smoothstep(0.0, 0.001, q);
      env *= smoothstep(0.3, 0.2, t);

      // { // highpass-like
      //   env *= exp(-50.0 * t);
      // }

      float phase = (
        42.0 * t
        - 7.0 * exp2(-t * 30.0)
        - 9.0 * exp2(-t * 90.0)
        - 9.0 * exp2(-t * 500.0)
      );
      // phase += 0.2 * exp2(-t * 20.0) * sin(4.0 * TAU * phase); // fm attack

      float wave = tanh(2.0 * sin(TAU * phase));

      dest += 0.4 * env * wave;
    }
  }

  { // hihat
    float t = tmod(time, S2T);
    float q = S2T - t;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp(-40.0 * t);

    vec2 wave = shotgun(4000.0 * t, 2.0, 0.0, 0.0);
    wave = tanh(1.5 * wave);

    dest += 0.4 * duck * env * wave;
  }

  { // open hihat
    float t = mod(time.x - 0.5 * B2T, B2T);
    float q = B2T - t;

    float env = exp2(-14.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      float odd = float(i % 2);
      float tt = (t + 0.3) * mix(1.0, 1.002, odd);
      vec3 dice = hash3f(vec3(i / 2));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * tt + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * tt + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * tt + dice2.zx);

      sum += wave * mix(1.0, 0.5, odd);
    }

    dest += 0.15 * env * duck * tanh(sum);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0008);
    float t = seq.t;

    float env = mix(
      exp2(-30.0 * t),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-100.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(1100.0 * t), 840.0 * t), 0.5, 2.0).xy;

    dest += 0.17 * tanh(20.0 * env * wave);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 8.0);

    float vel = fract(st * 0.78 + 0.05);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);

    float phase = 220.0 * t;
    phase += phase + 0.03 * sin(TAU * phase); // fm
    vec2 wave = shotgun(phase, 2.0, 0.2, exp2(mix(1.0, 3.0, vel)));

    dest += 0.16 * env * duck * tanh(8.0 * wave);
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-5.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 1.5, 0.0, 3.0);
    dest += 0.3 * mix(0.2, 1.0, duck) * env * tanh(8.0 * wave);
  }

  { // ambient stuff
    float t = time.z;

    vec3 noise = cyclic(vec3(1.1, 2.6, 0.8) * t, 0.5, 2.0);

    { // beep
      float amp = smoothstep(0.12, 0.13, noise.x);

      float tt = lofi(time.x, 1.0 / 12358.0);

      dest += 0.02 * amp * (
        cis(TAU * 9000.0 * t)
        + 0.5 * cheapnoise(256.0 * t)
      );
    }

    { // click
      float amp = smoothstep(0.32, 0.33, noise.y);

      float tt = tmod(time, S2T / 6.0);

      float env = exp(-200.0 * tt);

      dest += 0.02 * amp * env * (
        sin(TAU * 7000.0 * t)
      );
    }
  }

  { // bass
    vec2 sum = vec2(0.0);

    float l = S2T;
    float st = floor(time.z / l);
    float t = tmod(time, l);

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, l - t);

    float pitch = 36.0 + TRANSPOSE + CHORD_NOTE(0);

    { // sub
      float freq = p2f(pitch);
      float phase = freq * t;

      float wave = sin(TAU * phase);

      sum += env * wave;
    }

    repeat(i, 8) { // unison saw
      float fi = float(i);
      vec3 dice = hash3f(vec3(i, 21, 55));
      dice = hash3f(dice);

      float pitcht = pitch;
      pitcht += 0.4 * u2b(fi / 7.0);
      float freq = p2f(pitcht);
      float phase = freq * t + dice.y;

      vec2 wave = vec2(0.0);
      float k = 0.99 - 0.4 * t;
      wave += vec2(cheapfiltersaw(phase, k)) * rotate2D(fi + 0.8);

      sum += env * wave / 4.0;
    }

    dest += 0.4 * duck * sum;
  }

  { // pad
    vec2 sum = vec2(0.0);

    repeat(i, 64) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(i, 88, 72));
      vec2 dicen = boxMuller(dice.xy);
      dice = hash3f(dice);

      float l = 16.0 * B2T;
      float tt = tmod(time - l * dice.x, 16.0 * B2T);
      float fade = sin(PI * tt / l);

      float pitch = 48.0 + TRANSPOSE + CHORD_NOTE(i);
      pitch += 0.1 * dicen.x;
      float freq = p2f(pitch);
      float phase = freq * tt + dice.y;
      phase += 0.001 * freq * sin(8.0 * tt + TAU * dice.z);

      vec2 wave = vec2(0.0);
      wave += 0.2 * cheapfiltersaw(phase, 0.9);
      wave += 0.3 * cheapfiltersaw(2.0 * phase, 0.9);
      vec3 p = vec3(cis(TAU * phase), 2.0);
      wave += cyclic(p, 0.5, 1.8).xy;
      p = vec3(cis(2.0 * TAU * phase), 5.0);
      p.yz *= rotate2D(2.0 * sin(tt));
      wave += 0.6 * cyclic(p, 0.5, 4.0).xy;
      wave *= rotate2D(fi);

      sum += fade * wave / 32.0;
    }

    dest += 0.4 * mix(0.1, 1.0, duck) * tanh(4.0 * sum);
  }

  { // arp
    float l = S2T;

    vec2 sum = vec2(0.0);

    repeat(i, 4) {
      float fi = float(i);

      float off = -3.0 * S2T * fi;
      vec4 timet = mod(time - off, timeLength);
      float st = floor(timet.z / l);
      float t = mod(timet.x, l);

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, l - t);
      env *= exp(-1.0 * t);
      float delaydecay = exp(-fi);

      float ni = floor(mod(3.08 * st, 14.0));
      float pitch = 48.0 + TRANSPOSE + CHORD_NOTE(2.0 + mod(ni, 7.0));
      pitch += 12.0 * floor(ni / 7.0);
      float freq = p2f(pitch);
      float phase = freq * t;

      vec2 wave = vec2(0.0);
      wave += step(mix(0.4, 0.6, b2u(sin(9.0 * timet.z))), fract(phase + vec2(0.0, 0.3))) * 2.0 - 1.0;

      sum += delaydecay * env * wave;
    }

    dest += 0.1 * mix(0.2, 1.0, duck) * sum;
  }

  return clip(1.3 * tanh(dest));
}
