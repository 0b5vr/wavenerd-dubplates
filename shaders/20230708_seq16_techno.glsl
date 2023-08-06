#define BPM bpm

#define B2T (60.0 / bpm)
#define p2f(i) (exp2(((i)-69.0) / 12.0) * 440.0)
#define screen(a, b) (1.0 - (1.0 - (a)) * (1.0 - (b)))
#define saturate(i) clamp(i, 0.0, 1.0)
#define linearstep(a, b, x) saturate(((x) - (a)) / ((b) - (a)))
#define repeat(i, n) for (int i = 0; i < (n); i++)

const float TRANSPOSE = 4.0;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;
const uint UINT_MAX = 0xffffffffu;

const int st = 0x8888;

uvec3 pcg3d(uvec3 v) {
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

vec3 pcg3df(vec3 v) {
  uvec3 r = pcg3d(floatBitsToUint(v));
  return vec3(r) / float(UINT_MAX);
}

vec2 cis(float t) { return vec2(cos(t), sin(t)); }

mat2 rotate2D(float t) { return mat2(cos(t), sin(t), -sin(t), cos(t)); }

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
}

vec2 shotgun(float t, float spread, float snap) {
  vec2 sum = vec2(0);
  repeat(i, 64) {
    vec3 dice = pcg3df(vec3(i));

    float partial = exp2(spread * dice.x);
    partial = mix(partial, floor(partial + .5), snap);

    sum += vec2(sin(TAU * t * partial)) * rotate2D(TAU * dice.y);
  }
  return sum / 64.;
}

mat3 getOrthogonalBasis(vec3 z) {
  z = normalize(z);
  vec3 up = abs(z.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 x = normalize(cross(up, z));
  return mat3(x, cross(z, x), z);
}

vec3 cyclicNoise(vec3 p, float lacu, float pers) {
  vec4 sum = vec4(0.0);
  mat3 basis = getOrthogonalBasis(vec3(-1.0, 2.0, -3.0));
  float amp = 1.0;

  repeat(i, 5) {
    p *= basis;

    p += sin(p.yzx);
    sum += amp * vec4(cross(cos(p), sin(p.zxy)), 1.0);

    amp *= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
}

vec4 seq16(int seq, float st) {
  int sti = int(st);
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - prevStepBehind;
  float nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + nextStepForward;

  return vec4(prevStep, st - prevStep, nextStep, nextStep - st);
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float sidechain = 1.0;

  { // kick
    vec4 seq = seq16(0x8888, time.y / (0.25 * B2T));
    float s = seq.s;
    float t = seq.t * (0.25 * B2T);
    float q = seq.q * (0.25 * B2T);

    float env = linearstep(0.3, 0.1, t) * linearstep(0.0, 0.01, q);

    dest += 0.4 * env * tanh(2.0 * sin(
      320.0 * t
      - 65.0 * exp(-30.0 * t)
      - 40.0 * exp(-500.0 * t)
    ));

    sidechain = smoothstep(0.0, 0.3, t) * linearstep(0.0, 0.01, q);
  }

  { // hihat
    vec4 seq = seq16(0xffff, time.y / (0.25 * B2T));
    float s = seq.s;
    float t = seq.t * (0.25 * B2T);
    float q = seq.q * (0.25 * B2T);

    float env = exp(-60.0 * t);
    dest += 0.2 * env * mix(1.0, sidechain, 0.8) * tanh(8.0 * shotgun(4000.0 * t, 1.8, 0.0));
  }

  { // ride
    vec4 seq = seq16(0xaaaa, time.y / (0.25 * B2T));
    float s = seq.s;
    float t = seq.t * (0.25 * B2T);
    float q = seq.q * (0.25 * B2T);

    float env = exp(-10.0 * t);
    dest += 0.2 * env * mix(1.0, sidechain, 0.8) * tanh(8.0 * shotgun(3000.0 * t, 2.3, 0.5));
  }

  { // clap
    vec2 sum = vec2(0.0);
    vec4 seq = seq16(0x0808, time.y / (0.25 * B2T));
    float s = seq.s;
    float t = seq.t * (0.25 * B2T);
    float q = seq.q * (0.25 * B2T);

    float env = mix(exp(-30.0 * t), exp(-200.0 * mod(t, 0.013)),
                    exp(-80.0 * max(0.0, t - 0.02)));

    vec2 wave =
        cyclicNoise(vec3(4.0 * cis(1200.0 * t), 440.0 * t), 2.0, 0.7).xy;
    dest += 0.14 * tanh(20.0 * env * wave);
  }

  { // bass
    vec2 sum = vec2(0.0);
    vec4 seq = seq16(0xadad, time.y / (0.25 * B2T));
    float s = seq.s;
    float t = seq.t * (0.25 * B2T);
    float q = seq.q * (0.25 * B2T);

    float freq = p2f(24.0 + TRANSPOSE);
    float phase = freq * t;

    float env = linearstep(0.0, 0.01, t) * linearstep(0.0, 0.01, q - 0.03);
    vec2 wave = vec2(sin(TAU * phase));

    vec3 pos = vec3(5.0 * cis(TAU * phase), 44.0);
    wave += 0.5 * tanh(5.0 * cyclicNoise(pos, 1.2, 0.5).xy);

    dest += 0.3 * mix(1.0, sidechain, 1.0) * env * wave;
  }

  { // pad
    vec2 sum = vec2(0.0);
    vec4 seq = seq16(0x8926, time.y / (0.25 * B2T));
    float s = seq.s;
    float t = seq.t * (0.25 * B2T);
    float q = seq.q * (0.25 * B2T);
    
    vec3 dices = pcg3df(vec3(8.0 + s));

    const int CHORD[] = int[](0, 7, 10, 14, 15, 17);

    repeat(i, 48) {
      float fi = float(i);
      vec3 dicei = pcg3df(vec3(fi));
      vec2 dicein = boxMuller(dicei.xy);

      float freq = p2f(48.0 + TRANSPOSE + float(CHORD[i % 6]) + 0.1 * dicein.x);
      float phase = freq * t + dicei.z;

      float env = linearstep(0.0, 0.01, t) * linearstep(0.0, 0.02, q - 0.02);
      vec3 pos = vec3(2.0 * cis(TAU * phase), 0.0) + 10.0 * dices;
      vec2 wave = cyclicNoise(pos, 2.0, 1.0 * exp(-5.0 * t)).xy * rotate2D(fi);

      sum += env * wave / 24.0;
    }

    dest += 0.9 * mix(1.0, sidechain, 0.2) * sum;
  }

  return tanh(dest);
}
