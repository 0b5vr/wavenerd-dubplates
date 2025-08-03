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
#define TRANSPOSE -3.0

const float SWING = 0.55;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;

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

vec3 corruptor(float t, float corrupt) {
  const float PERS = 0.5;
  const float LACU = 1.4;

  float n = cyclic(vec3(t, corrupt, 0.0), PERS, LACU).x;
  n = floor(2.0 * n + corrupt);
  return hash3f(vec3(t, 0.0, n));
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

      // { // highpass-like
      //   env *= exp(-50.0 * t);
      // }

      float phase = (
        40.0 * t
        - 10.0 * exp2(-t * 18.0)
        - 3.0 * exp2(-t * 80.0)
      );
      float phaset = 4.0 * exp2(-t * 500.0);

      float wave = sin(TAU * phase);
      wave += sin(TAU * phaset);
      wave = tanh(2.0 * wave);

      dest += 0.5 * env * wave;
    }
  }

  { // bass
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.y;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q - 0.1 * S2T);

    float pitch = TRANSPOSE + 36.0;
    float freq = p2f(pitch);
    float phase = freq * t;

    // fm
    phase += 0.2 * exp2(-40.0 * t) * sin(2.0 * TAU * phase);

    float wave = sin(TAU * phase);
    wave = tanh(wave);

    dest += 0.4 * env * mix(0.0, 1.0, duck) * wave;
  }

  { // hihat
    vec4 seq = seq16(time.y, 0x7777);
    float t = seq.y;

    float env = exp2(-120.0 * t);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-10.0 * t) * sin(wave + exp2(14.30 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp2(-10.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp2(-10.0 * t) * sin(wave + exp2(12.92 + 0.2 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.17 * env * mix(0.2, 1.0, duck) * tanh(2.0 * sum);
  }

  { // open hihat
    float t = mod(time.x - 0.5 * B2T, B2T);
    float q = B2T - t;

    float env = exp2(-15.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      float odd = float(i % 2);
      float tt = (t + 0.3) * mix(1.0, 1.002, odd);
      vec3 dice = hash3f(vec3(i / 2));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.00 + 0.1 * dice.x) * tt + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.71 + 0.2 * dice.y) * tt + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(13.52 + 0.5 * dice.z) * tt + dice2.zx);

      sum += wave * mix(1.0, 0.5, odd);
    }

    dest += 0.26 * env * duck * tanh(sum);
  }

  { // snare909
    vec4 seq = seq16(time.y, 0x9479);
    float t = seq.t;
    float q = seq.q;

    float env = exp(-20.0 * max(t - 0.04, 0.0)) * smoothstep(0.0, 0.01, q);

    float bodyphase = 220.0 * t - 4.0 * exp2(-t * 200.0);
    float noisephase = 600.0 * t;
    vec2 wave = mix(
      mix(
        cis(1.5 * TAU * bodyphase),
        cis(TAU * bodyphase),
        0.6
      ),
      cheapnoise(128.0 * t) - cheapnoise(128.0 * t - 0.008),
      0.3
    );

    dest += 0.3 * mix(0.1, 1.0, duck) * tanh(4.0 * env * wave);
  }

  { // ride
    vec4 seq = seq16(time.y, 0xaaaa);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.002, q);
    env *= exp2(-6.0 * t);

    vec2 wave = shotgun(2800.0 * t, 1.8, 0.0, 0.8);
    wave = tanh(2.0 * wave);

    dest += 0.26 * mix(0.1, 1.0, duck) * env * wave;
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 0.0);
    dest += 0.4 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // corruptor synth
    vec2 sum = vec2(0.0);

    repeat(iDelay, 4) {
      float fiDelay = float(iDelay);
      float delaydecay = exp(-fiDelay);
      float toff = 3.0 * S2T * fiDelay;

      vec4 seq = seq16(time.y - toff, 0xffff);
      float st = mod(floor((time.z - toff) / S2T), 8.0);
      float t = seq.t;
      float q = seq.q;

      float corrupt = 20.0; // CORRUPTOR SEED HERE
      // corrupt = 0.04 * lofi(time.w, B2T); // auto corruptor
      vec3 dicec = corruptor(2.0 * st, corrupt);
      vec3 dicec2 = corruptor(22.4 + 4.5 * st, corrupt);

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, 0.5 * q);
      env *= step(dicec.x, 0.7); // gate

      float pitch = TRANSPOSE + mix(40.0, 70.0, dicec2.x);
      float freq = p2f(pitch);
      float phase = freq * t;

      float modamp = exp2(mix(-2.0, 1.0, dicec2.y));
      float modmul = exp2(mix(0.0, 3.0, dicec2.z));
      phase += modamp / modmul * sin(TAU * modmul * phase); // fm

      vec2 wave = vec2(0.0);
      wave += cis(TAU * phase); // osc 1
      wave *= exp2(-exp2(mix(1.0, 5.0, dicec.y)) * t); // decay
      wave *= exp2(mix(0.0, 1.5, dicec.z)); // amp
      wave = tri(wave / 4.0); // fold

      sum += delaydecay * env * wave;
    }

    dest += 0.2 * mix(0.0, 1.0, duck) * sum;
  }

  { // fx synth
    vec2 sum = vec2(0.0);
    repeat(i, 8) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi, 22, 30));

      float l = 4.0 * B2T;
      float t = tmod(time, l);

      vec3 muls = vec3(4.59, 4.04, 0.5) * exp2(0.005 * (dice - 0.5));
      float phase = 120.0 * t;
      float op2 = 2.0 * t / l * sin(muls.y * TAU * phase);
      float op3 = 0.2 * sin(muls.z * TAU * phase);
      float op1 = t / l * sin(muls.x * TAU * phase + TAU * op2 + TAU * op3);

      sum += vec2(op1 / 4.0) * rotate2D(TAU / 8.0 * fi);
    }

    dest += 0.1 * mix(0.1, 1.0, duck) * sum;
  }

  return clip(1.3 * tanh(dest));
}
