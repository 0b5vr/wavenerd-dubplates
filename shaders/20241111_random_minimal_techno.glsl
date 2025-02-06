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

    float env = smoothstep(0.3, 0.1, t) * smoothstep(0.0, 0.001, q);

    // { // hi-pass like
    //   env *= exp(-70.0 * t);
    // }

    float tt = t;
    float wave = sin(TAU * (
      48.0 * tt
      - 4.0 * exp2(-tt * 20.0)
      - 3.5 * exp2(-tt * 90.0)
      - 2.0 * exp2(-tt * 300.0)
    ));
    dest += 0.5 * tanh(2.7 * env * wave);
  }

  { // bass
    vec4 seq = seq16(time.y, 0xffff);
    float s = mod(seq.s + floor(time.z / 16.0 / S2T), 5.0);
    float t = seq.t;
    float q = seq.q;

    float env = linearstep(0.0, 0.001, t) * linearstep(0.0, 0.01, q) * exp(-5.0 * t);

    float pitch = 36.0 + 5.0 * sin(2.0 * s) + TRANSPOSE;
    float freq = p2f(pitch);
    vec2 phase = vec2(freq * t);
    phase += 0.1 * exp(-10.0 * t) * cis(3.0 * TAU * freq * t + time.z);
    vec2 wave = vec2(sin(TAU * phase));

    dest += 0.4 * sidechain * env * wave;
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;

    float vel = fract(seq.s * 0.611);
    float env = exp2(-exp2(7.0 - 1.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.0);
    dest += 0.24 * env * mix(0.2, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.t;
    float q = seq.q;

    vec2 sum = vec2(0.0);

    float env = linearstep(0.0, 0.01, q) * mix(
      exp2(-40.0 * t),
      exp2(-1.0 * t),
      0.04
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

    dest += 0.3 * env * mix(0.2, 1.0, sidechain) * tanh(2.0 * sum);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0101);
    float t = seq.t;

    float env = mix(
      exp2(-70.0 * t),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-120.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0).xy;

    dest += 0.18 * tanh(20.0 * env * wave);
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
      wave = 2.7 * env * sin(wave + exp2(13.67 + 0.1 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(13.19 + 0.9 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.05 * env * mix(0.3, 1.0, sidechain) * tanh(sum);
  }

  // { // snare909
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.y;

  //   float env = exp(-20.0 * t);

  //   vec2 wave = (
  //     cyclic(vec3(cis(4000.0 * t), 4000.0 * t), 1.0, 2.0).xy
  //     + sin(1400.0 * t - 40.0 * exp2(-t * 200.0))
  //   );

  //   dest += 0.3 * mix(0.3, 1.0, sidechain) * tanh(4.0 * env * wave);
  // }

  { // perc
    vec4 seq = seq16(time.y, 0x0020);
    float t = seq.t;

    float env = mix(
      exp2(-20.0 * t),
      exp2(-1.0 * t),
      0.1
    );

    float freq = 720.0;
    vec2 phase = vec2(0.0);
    phase += TAU * t * freq;
    phase += sin(0.262 * phase);
    phase += exp2(-4.0 * t) * 0.5 * sin(7.27 * phase);
    vec2 wave = sin(phase);

    dest += 0.2 * sidechain * tanh(2.0 * env * wave);
  }

  { // modshit
    vec2 sum = vec2(0.0);

    repeat(i, 3) {
      float fi = float(i);
      float delayoff = 2.0 * S2T * fi;
      float delaydecay = exp2(-fi);

      vec4 seq = seq16(time.y - delayoff, 0xffff);
      float t = seq.t;
      float q = seq.q;

      vec3 dice = hash3f(mod(seq.sss, 8.0) + vec3(1421, 140, 1224)); // random shit

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp2(-exp2(2.0 + 4.0 * dice.z) * t);

      vec2 wave = vec2(0.0);
      repeat(i, 4) {
        vec3 dice2 = hash3f(dice + float(i));
        vec2 phase = vec2(t * 120.0 * exp2(2.0 * dice.x) + dice2.xy);
        phase = fract(phase - 0.5 * fract(t * 400.0 * exp2(2.0 * dice.y) + dice2.xy));
        phase = fract(phase - 0.1 * fract(t * 700.0 * exp2(2.0 * dice.z) + dice2.xy));
        wave += (2.0 * phase - 1.0) * rotate2D(TAU * dice2.z);
      }

      sum += delaydecay * env * tanh(4.0 * wave);
    }

    dest += 0.1 * mix(0.2, 1.0, sidechain) * sum;
  }

  return clip(1.3 * tanh(dest));
}
