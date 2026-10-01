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

uniform vec4 param_knob3; // kick cut
uniform vec4 param_knob4; // riff fm ratio
uniform vec4 param_knob5; // riff lofi

#define p3 paramFetch(param_knob3)
#define p4 paramFetch(param_knob4)
#define p5 paramFetch(param_knob5)

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

float glidephase(float t, float t1, float pitch0, float pitch1) {
  if (pitch0 == pitch1) {
    return t * p2f(pitch1);
  }

  float m0 = (pitch0 - 69.0) / 12.0;
  float m1 = (pitch1 - 69.0) / 12.0;
  float b = (m1 - m0) / t1;

  return (
    + p2f(pitch0) * (pow(2.0, b * min(t, t1)) - 1.0) / b / LN2
    + max(0.0, t - t1) * p2f(pitch1)
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

    {
      float env = smoothstep(0.0, 0.001, q);
      env *= smoothstep(0.3, 0.1, t);
      env *= mix(1.0, exp2(-60.0 * t), p3); // hpf-like

      float phase = (
        40.0 * t
        - 3.0 * exp2(-t * 20.0)
        - 3.0 * exp2(-t * 80.0)
      );
      phase += 0.1 * exp2(-t * 20.0) * sin(4.0 * TAU * phase); // fm attack

      float wave = tanh(sin(TAU * phase));

      dest += 0.7 * env * wave;
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

    dest += 0.5 * (1.0 - p3) * mix(0.0, 1.0, duck) * env * wave;
  }

  { // low freq noise
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

    vec2 wave = vec2(0.0);
    wave += 0.8 * cheapnoise(2.0 * t);
    wave += 0.1 * cheapnoise(8.0 * t);
    wave += 0.03 * cheapnoise(16.0 * t);
    wave = mix(wave, 0.5 * vec2(wave.x + wave.y), 0.3);

    dest += 0.2 * duck * env * wave;
  }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-100.0 * t);

  //   vec2 wave = shotgun(3100.0 * t, 2.0, 0.0, 0.0);
  //   wave = tanh(6.0 * wave);

  //   dest += 0.3 * mix(0.1, 1.0, duck) * env * wave;
  // }

  // { // shaker
  //   float t = mod(time.x, S2T);
  //   float st = mod(floor(time.y / S2T), 8.0);

  //   float vel = fract(st * 0.42 + 0.43);
  //   float env = smoothstep(0.0, 0.02, t) * exp(-exp2(7.0 - 4.0 * vel) * t);
  //   vec2 wave = cyclic(vec3(cis(3800.0 * t), exp2(8.0 + 4.0 * vel) * t), 0.8, 2.0).xy;
  //   dest += 0.2 * env * duck * tanh(2.0 * wave);
  // }

  // { // open hihat
  //   vec4 seq = seq16(time.y, 0x2222);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-40.0 * t);

  //   vec2 wave = shotgun(3100.0 * t, 1.2, 0.1, 2.0);
  //   wave = tanh(3.0 * env * wave);

  //   dest += 0.5 * mix(0.1, 1.0, duck) * wave;
  // }

  // { // ride
  //   vec4 seq = seq16(time.y, 0xaaaa);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = exp2(-4.0 * t);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 4.5 * env * sin(wave + exp2(12.60 + 0.1 * dice.x) * t + dice2.xy);
  //     wave = 2.2 * env * sin(wave + exp2(14.87 + 0.5 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * env * sin(wave + exp2(14.29 + 1.0 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.06 * env * duck * tanh(sum);
  // }

  // { // clap
  //   vec4 seq = seq16(time.y, 0x6060);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = mix(
  //     exp2(-80.0 * t),
  //     exp2(-500.0 * mod(t, 0.012)),
  //     exp2(-100.0 * max(0.0, t - 0.02))
  //   );

  //   vec2 wave = cyclic(vec3(4.0 * cis(1100.0 * t), 1540.0 * t), 0.5, 2.0).xy;

  //   dest += 0.12 * mix(0.5, 1.0, duck) * tanh(20.0 * env * wave);
  // }

  // { // crash
  //   float t = mod(time.z, 64.0 * B2T);

  //   float env = mix(exp(-t), exp(-10.0 * t), 0.7);
  //   vec2 wave = shotgun(4500.0 * t, 1.4, 0.0, 1.0);
  //   dest += 0.4 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  // }

  { // sub riff
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s;
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);
    float stmod = fract(0.655 * st + 0.44);
    float cutenv = smoothstep(0.0, 0.01, t) * exp2(-10.0 * t);
    float cutoff = exp2(
      6.0
      + 2.0 * stmod
      + 4.0 * cutenv
    );

    vec2 sum = vec2(0.0);
    repeat(iPartial, 128) {
      float partial = 1.0 + float(iPartial);
      partial = pow(partial, 1.1);
      partial = mix(partial, 1.0, 0.04);

      const float basefreq = 50.0;
      float freq = basefreq * partial;

      vec2 lpf = ladderLPF(freq, cutoff, 0.3);
      vec2 hpf = twoPoleHPF(freq, 1000.0, 0.0);

      vec2 phase = vec2(t * freq);
      vec2 wave = sin(TAU * phase + lpf.y + hpf.y) / partial * lpf.x * hpf.x;

      sum += env * wave * rotate2D(2.4 * float(iPartial));
    }

    dest += 0.16 * mix(0.2, 1.0, duck) * tanh(5.0 * sum);
  }

  { // riff
    vec2 sum = vec2(0.0);

    const float GLIDE = 0.012;

    repeat(i, 4) {
      float fi = float(i);
      float delaydecay = exp(-fi);

      vec4 seq = seq16(time.y - 3.0 * S2T * fi, 0xffff);
      float st = seq.s + lofi(time.z / S2T, 16.0);
      float t = seq.t;
      float q = seq.q;

      vec3 dice0 = hash3f(vec3(mod(st - 1.0, 8.0), 12, 2));
      vec3 dice1 = hash3f(vec3(mod(st, 8.0), 12, 2));

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

      float pitch0 = mix(44.0, 90.0, dice0.y) + TRANSPOSE;
      float pitch1 = mix(44.0, 90.0, dice1.y) + TRANSPOSE;
      vec2 phase = vec2(glidephase(t, GLIDE, pitch0, pitch1));

      // stereo
      phase += mix(dice0.xy, dice1.xy, saturate(t / GLIDE));

      // reroll
      dice0 = hash3f(vec3(mod(st - 1.0, 7.0), 1, 4));
      dice1 = hash3f(vec3(mod(st, 7.0), 1, 4));

      // lofi
      float lofiamp0 = step(0.5, dice0.x);
      float lofiamp1 = step(0.5, dice1.x);
      float lofiamp = mix(lofiamp0, lofiamp1, saturate(t / GLIDE));
      phase = mix(phase, lofi(phase, exp2(-6.0 + 4.0 * p5)), lofiamp);

      // fm
      float fmamp0 = smoothstep(0.5, 1.0, dice0.y);
      float fmamp1 = smoothstep(0.5, 1.0, dice1.y);
      float fmamp = mix(fmamp0, fmamp1, saturate(t / GLIDE));
      // phase += fmamp * sin(2.5 * TAU * phase);
      phase += fmamp * sin(mix(1.25, 3.5, p4) * TAU * phase);

      vec2 wave = vec2(sin(TAU * phase));

      sum += env * delaydecay * wave / 4.0;
    }

    dest += 0.4 * mix(0.3, 1.0, duck) * tanh(sum);
  }

  { // oidos drone
    vec2 sum = vec2(0.0);

    repeat(i, 2500) {
      vec3 diceA = hash3f(vec3(i / 50, 35, 12));
      vec3 diceB = hash3f(vec3(i, 35, 12));

      float t = mod(time.z - diceA.x * (64.0 * B2T), 64.0 * B2T);
      float env = sin(PI * t / (64.0 * B2T));

      float tone = 7.0 + 8.0 * diceA.y + 0.05 * diceB.y;
      float freq = exp2(tone);
      vec2 phase = t * freq + fract(diceB.xy * 999.0);

      float amp = 1.0;
      const float FILTER_WIDTH = 0.3;

      float pm = 0.2;
      phase += pm * fract(4.0 * phase); // add high freq

      sum += amp * env * sin(TAU * phase) / 1000.0;
    }

    dest += 0.4 * mix(0.3, 1.0, duck) * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  dest *= 1.1;
  return dest;
}
