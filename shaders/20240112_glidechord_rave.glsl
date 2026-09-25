#define TRANSPOSE 5.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float SWING = 0.52;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

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

vec2 cheapfiltersaw(vec2 phase, float k) {
  vec2 wave = fract(phase);
  vec2 c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
}

vec2 mainAudioDry(vec4 time) {
  vec2 dest = vec2(0);
  float duck;

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    float wave = sin(
      240.0 * t
      - 60.0 * exp2(-t * 30.0)
      - 10.0 * exp2(-t * 90.0)
      - 10.0 * exp2(-t * 300.0)
    );
    dest += 0.5 * env * clip(1.5 * wave);
  }

  { // bass
    vec4 seq = seq16(time.y, 0x6d6d);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp(-5.0 * t);

    float note = TRANSPOSE + 24.0;
    float phase = p2f(note) * t;

    vec2 wave = vec2(sin(TAU * phase));
    wave += tanh(3.0 * sin(
      TAU * phase
      + 0.8 * env * cis(200.0 * TAU * exp(-3.0 * t))
    ));

    dest += 0.4 * env * duck * wave;
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s;
    float t = seq.t;
    float q = seq.q;

    float vel = fract(st * 0.2 + 0.42);
    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp(-exp2(7.0 - 3.0 * vel) * t);

    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.2 * env * mix(0.2, 1.0, duck) * tanh(8.0 * wave);
  }

  { // open hihat
    float t = mod(time.x - 0.5 * B2T, B2T);
    float q = B2T - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp2(-10.0 * t);

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      float odd = float(i % 2);
      float tt = (t + 0.3) * mix(1.0, 1.002, odd);
      vec3 dice = hash3f(vec3(i / 2));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.36 + 0.1 * dice.x) * tt + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.88 + 0.3 * dice.y) * tt + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(15.02 + 0.2 * dice.z) * tt + dice2.zx);

      sum += wave * mix(1.0, 0.5, odd);
    }

    dest += 0.18 * duck * env * tanh(sum);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0808);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= mix(
      exp(-30.0 * t),
      exp(-200.0 * mod(t, 0.013)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 1940.0 * t), 0.5, 3.0).xy;

    dest += 0.14 * tanh(20.0 * env * wave);
  }

  { // perc
    vec4 seq = seq16(time.y, 0x0808);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp2(-5.0 * t);

    vec2 osc = shotgun(1400.0 * t, 1.5, 0.0, 0.0);

    dest += 0.1 * env * tanh(5.0 * osc);
  }

  { // ride
    vec4 seq = seq16(time.y, 0xaaaa);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp(-5.0 * t);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * env * sin(wave + exp2(12.10 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * env * sin(wave + exp2(14.87 + 0.1 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(13.89 + 0.1 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.07 * env * mix(0.2, 1.0, duck) * tanh(sum);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.41 + 0.63);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);
    vec2 wave = cyclic(vec3(cis(200.0 * t), exp2(8.0 + 3.0 * vel) * t), 1.0, 2.0).xy;
    dest += 0.15 * env * mix(0.3, 1.0, duck) * tanh(2.0 * wave);
  }

  { // crash
    float t = time.z;

    float env = mix(exp2(-t), exp2(-14.0 * t), 0.7);
    vec2 wave = shotgun(4000.0 * t, 2.5, 0.0, 0.0);
    dest += 0.4 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // pad
    vec2 sum=vec2(0);

    const int pitchTable[8] = int[](0, 3, 7, 10, 12, 14, 17, 19);

    repeat(i, 64) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi));

      float t = mod(time.z, 8.0 * B2T);

      float note = TRANSPOSE + 48.0 + float(pitchTable[i % 8]);
      float detune = exp2(0.006 * boxMuller(dice.xy).x);
      float phase = glidephase(t - 0.9 * B2T, 0.5 * B2T, note + 6.0, note) * detune + dice.z;

      vec3 origin = vec3(3.0, 3.0, 5.0) + 0.4 * t;
      vec3 dir = vec3(3.0, 8.0, -2.0);
      vec3 p = vec3(origin + dir * fract(phase)) + 8.5;
      vec2 wave = cyclic(p, 0.6, 2.0).xy;

      sum += wave * rotate2D(fi) / 24.0;
    }

    dest += 0.9 * mix(0.3, 1.0, duck) * tanh(sum);
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  dest = mainAudioDry(time);

  return clip(1.3 * tanh(dest));
}
