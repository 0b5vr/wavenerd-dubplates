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
uniform vec4 param_knob4; // riff fm

#define p3 paramFetch(param_knob3)
#define p4 paramFetch(param_knob4)

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
      env *= smoothstep(0.3, 0.2, t);
      env *= mix(1.0, exp2(-50.0 * t), p3); // hpf-like

      float phase = (
        47.0 * t
        - 3.0 * exp2(-t * 30.0)
        - 4.0 * exp2(-t * 80.0)
      );
      phase += 0.1 * exp2(-t * 20.0) * sin(4.0 * TAU * phase); // fm attack

      float wave = tanh(2.0 * sin(TAU * phase));

      dest += 0.6 * env * wave;
    }
  }

  { // sub kick
    vec4 seq = seq16(time.y, 0x3333);
    float t = seq.t;
    float q = seq.q;

    {
      float env = smoothstep(0.0, 0.001, q);
      env *= smoothstep(0.3, 0.2, t);

      float phase = (
        48.0 * t
        - 2.0 * exp2(-t * 60.0)
      );

      float wave = sin(TAU * phase);

      dest += 0.4 * (1.0 - p3) * duck * env * wave;
    }
  }

  { // low freq noise
    float t = time.x;

    vec2 wave = vec2(sin(48.0 * TAU * t));
    wave += 0.4 * cheapnoise(2.0 * t);
    wave += 0.1 * cheapnoise(8.0 * t);
    wave += 0.01 * cheapnoise(32.0 * t);
    wave = mix(wave, 0.5 * vec2(wave.x + wave.y), 0.3);

    dest += 0.2 * duck * wave;
  }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp(-50.0 * t);

  //   vec2 wave = shotgun(4700.0 * t, 1.8, 0.0, 0.0);
  //   wave = tanh(1.5 * wave);

  //   dest += 0.4 * mix(0.1, 1.0, duck) * env * wave;
  // }

  // { // open hihat
  //   vec4 seq = seq16(time.y, 0x1222);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp(-10.0 * t);

  //   vec2 wave = shotgun(5100.0 * t, 2.5, 0.3, 1.0);
  //   wave = tanh(1.5 * wave);

  //   dest += 0.4 * mix(0.1, 1.0, duck) * env * wave;
  // }

  // { // shaker
  //   vec4 seq = seq16(time.y, 0x9252);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.02, t) * smoothstep(0.0, 0.01, q);
  //   env *= exp(-10.0 * t);

  //   vec2 wave = shotgun(3200.0 * t, 1.8, 0.3, 1.0);
  //   wave = tanh(1.5 * wave);

  //   dest += 0.24 * mix(0.1, 1.0, duck) * env * wave;
  // }

  // { // clap
  //   vec4 seq = seq16(time.y, 0x2001);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = mix(
  //     exp2(-60.0 * t),
  //     exp2(-500.0 * mod(t, 0.012)),
  //     exp2(-100.0 * max(0.0, t - 0.02))
  //   );

  //   vec2 wave = cyclic(vec3(4.0 * cis(1200.0 * t), 1540.0 * t), 0.5, 2.0).xy;

  //   dest += 0.14 * mix(0.3, 1.0, duck) * tanh(20.0 * env * wave);
  // }

  // { // ride
  //   vec4 seq = seq16(time.y, 0xaaaa);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  //   env *= exp(-5.0 * t);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 4.5 * env * sin(wave + exp2(11.80 + 0.2 * dice.x) * t + dice2.xy);
  //     wave = 2.2 * env * sin(wave + exp2(15.17 + 0.8 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * env * sin(wave + exp2(13.29 + 1.0 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.07 * mix(0.1, 1.0, duck) * env * tanh(sum);
  // }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4500.0 * t, 1.4, 0.0, 1.0);
    dest += 0.3 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // riff
    vec2 sum = vec2(0.0);

    repeat(i, 4) {
      float fi = float(i);
      float delaydecay = exp(-fi);

      vec4 seq = seq16(time.y - 3.0 * S2T * fi, 0xffff);
      float st = seq.s + lofi(time.z / S2T, 16.0);
      float t = seq.t;
      float q = seq.q;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

      vec3 dice0 = hash3f(vec3(mod(st - 1.0, 5.0), 0.82, 4.89));
      vec3 dice1 = hash3f(vec3(mod(st, 5.0), 0.82, 4.89));

      float p0 = mix(50.0, 90.0, dice0.x) + TRANSPOSE;
      float p1 = mix(50.0, 90.0, dice1.x) + TRANSPOSE;
      float phase = glidephase(t, 0.012, p0, p1);
      phase += 0.3 * p4 * sin(TAU * phase);

      vec2 wave = cis(TAU * phase);

      sum += env * delaydecay * wave / 4.0;
    }

    dest += 0.6 * mix(0.2, 1.0, duck) * tanh(sum);
  }

  { // pad
    vec2 sum = vec2(0.0);

    const int N_UNISON = 200;
    const int N_BUNDLE = 20;

    repeat(i, N_UNISON) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi, 55, 28));
      vec3 diceg = hash3f(vec3(floor(fi / float(N_BUNDLE)), 34, 58));

      float l = 16.0 * B2T;
      float toff = tmod(time - dice.x * l, l);

      float env = sin(PI * toff / l);

      float pitch = mix(7.0, 11.0, diceg.x) + 0.02 * dice.y;
      float freq = exp2(pitch);
      float phase = TAU * freq * toff;
      phase += 0.1 * sin(2.5 * TAU * phase);
      phase += 0.1 * sin(0.25 * TAU * phase);
      phase += 0.3 * sin(0.5 * TAU * phase);
      phase += TAU * dice.z;
      vec2 wave = cis(phase);

      sum += env * wave / float(N_UNISON / 2);
    }

    dest += 0.6 * mix(0.1, 1.0, duck) * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  dest *= 0.85;
  return dest;
}
