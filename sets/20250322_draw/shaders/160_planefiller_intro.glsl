#define TRANSPOSE 3.0
#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(i) clamp(i, -1., 1.)
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define lofi(i,m) (floor((i)/(m))*(m))

const float SWING = 0.64;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);
const float MIN3 = pow(2.0, 3.0 / 12.0);
const float P4 = pow(2.0, 5.0 / 12.0);
const float P5 = pow(2.0, 7.0 / 12.0);

uniform vec4 param_knob4; // choir length

#define p4 paramFetch(param_knob4)

// https://www.shadertoy.com/view/XlXcW4
vec3 hash3f(vec3 s) {
  uvec3 r = floatBitsToUint(s);
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  return vec3(r) / float(-1u);
}

vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D(float x) {
  vec2 v = cis(x);
  return mat2(v.x, v.y, -v.y, v.x);
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

vec2 shotgun(float t, float spread) {
  vec2 sum = vec2(0.0);

  repeat(i, 64) {
    vec3 dice = hash3f(float(i) + vec3(7, 1, 3));
    sum += vec2(sin(TAU * t * exp2(spread * dice.x))) * rotate2D(TAU * dice.y);
  }

  return sum / 64.0;
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 up = abs(z.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 x = normalize(cross(up, z));
  return mat3(x, cross(z, x), z);
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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  { // chord stuff
    const int N_CHORD = 8;
    const int CHORD[N_CHORD] = int[](
      0, 7, 10, 12, 15, 17, 19, 22
    );

    float t = time.z;
    float l = timeLength.z;
    float q = l - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    float trans = TRANSPOSE;

    { // choir
      vec2 sum = vec2(0.0);

      repeat(i, 64) {
        float fi = float(i);
        vec3 dice = hash3f(float(i) + vec3(8, 4, 2));

        float note = 48.0 + trans + float(CHORD[i % N_CHORD]);
        float freq = p2f(note) * exp2(0.016 * tan(2.0 * dice.y - 1.0));
        float phase = lofi(t * freq, 1.0 / 16.0);

        vec3 c = vec3(0.0);
        vec3 d = vec3(2.0, -3.0, -8.0);
        float k = 0.5;
        vec2 wave = cyclic(fract(phase) * d, k, 2.0).xy;

        sum += vec2(wave) * rotate2D(fi);
      }

      dest += 0.05 * env * sum;
    }
  }

  return clip(1.3 * tanh(0.9 * dest));
}
