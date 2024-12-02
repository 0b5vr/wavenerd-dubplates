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

const float SWING = 0.5;

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

float p2f(float p) {
  return exp2((p - 69.0) / 12.0) * 440.0;
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

vec3 cyclicNoise(vec3 p, float pump) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  repeat(i, 5) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum *= pump;
    p *= 2.;
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

float t2sSwing(float t) {
  float st = t / S2T;
  return 2.0 * floor(st / 2.0) + step(SWING, fract(0.5 * st));
}

float s2tSwing(float st) {
  return 2.0 * S2T * (floor(st / 2.0) + SWING * mod(st, 2.0));
}

vec4 seq16(float t, int seq) {
  t = mod(t, 16.0 * S2T);
  int sti = clamp(int(t2sSwing(t)), 0, 15);
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - prevStepBehind;
  float prevTime = s2tSwing(prevStep);
  float nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + nextStepForward;
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

  i = floor(floor(st + 1E-4) / interval + 1E-4);

  float prevStep = ceil(i * interval - 1E-4);
  float prevTime = s2tSwing(prevStep);
  float nextStep = ceil((i + 1.0) * interval - 1E-4);
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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float sidechain = 1.0;

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    sidechain = 0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    float tt = t;
    float wave = sin(
      250.0 * tt
      - 40.0 * exp(-tt * 10.0)
      - 20.0 * exp(-tt * 40.0)
      - 20.0 * exp(-tt * 200.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // bass
    vec4 seq = quant(time.z, 3.0);
    float t = seq.t;

    float env = linearstep(0.0, 0.001, t) * exp(-5.0 * t);

    float pitch = 36.0 + TRANSPOSE;
    float freq = p2f(pitch);
    vec2 phase = vec2(freq * t);
    phase += 0.3 * exp(-20.0 * t) * cis(4.0 * TAU * freq * t);
    phase += 0.04 * exp(-20.0 * t) * cis(14.0 * TAU * freq * t);
    vec2 wave = vec2(sin(TAU * phase));

    dest += 0.6 * mix(0.2, 1.0, sidechain) * env * wave;
  }

  { // hihat
    float st;
    vec4 seq = quant(time.y, 1.0, st);
    float t = seq.t;

    float vel = fract(st * 0.61 + 0.42);
    float env = exp(-exp2(6.0 - 2.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.1, 6.0);
    dest += 0.2 * env * sidechain * tanh(8.0 * wave);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0808);
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      exp(-50.0 * t),
      exp(-400.0 * mod(t, 0.012)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclicNoise(vec3(4.0 * cis(200.0 * t), 840.0 * t), 0.7).xy;

    dest += 0.12 * tanh(20.0 * env * wave);
  }

  { // ride
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.t;
    float q = seq.q;

    float env = exp(-4.0 * t) * smoothstep(0.0, 0.01, 2.0 * S2T - t);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 3.9 * env * sin(wave + exp2(12.30 + 0.3 * dice.x) * t + dice2.xy);
      wave = 2.8 * env * sin(wave + exp2(14.41 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(12.29 + 1.0 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.06 * env * sidechain * tanh(sum);
  }

  { // rim
    vec4 seq = quant(time.y, 1.3);
    float t = seq.t;
    float q = seq.q;

    float env = step(0.0, t) * exp(-300.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * wave * vec2(1, -1);
  }

  { // lead
    const int N_NOTES = 16;
    const float NOTES[N_NOTES] = float[](
      -12.0, 0.0, -12.0, 2.0,
      3.0, -12.0, 14.0, 19.0,
      0.0, 22.0, 17.0, -12.0,
      0.0, 7.0, -12.0, 19.0
    );

    vec2 sum = vec2(0.0);
    repeat(i, 8) {
      float fi = float(i);

      repeat(j, 4) {
        float fj = float(j);
        vec3 dice = hash3f(vec3(fi, fj, 1));
        vec2 dicen = boxMuller(dice.xy);

        float l = 0.8 * S2T;
        float t = mod(time.y - s2tSwing(fi), 16.0 * S2T);
        int noteIndex = i + (t > 8.0 * S2T ? 8 : 0);
        t = mod(t, 8.0 * S2T);

        float delayDecay = exp(-floor(t / (3.0 * S2T)));
        t = mod(t, 3.0 * S2T);
        float q = 0.5 * S2T - t;
        float env = linearstep(0.0, 0.01, t) * exp(-10.0 * max(0.0, -q));

        float detune = 0.15 * dicen.x;
        float p0 = 48.0 + TRANSPOSE + NOTES[(noteIndex + N_NOTES - 1) % N_NOTES] + detune;
        float p1 = 48.0 + TRANSPOSE + NOTES[noteIndex] + detune;
        float phase = glidephase(t, 0.3 * S2T, p0, p1);
        phase += dice.z;

        vec2 wave = vec2(
          + cheapfiltersaw(1.0 * phase, 0.99)
          - cheapfiltersaw(1.0 * phase, 0.8)
        );
        sum += env * delayDecay * wave * rotate2D(TAU * fj / 4.0);
      }
    }

    dest += 0.2 * mix(0.2, 1.0, sidechain) * sum;
  }

  return clip(1.3 * tanh(dest));
}
