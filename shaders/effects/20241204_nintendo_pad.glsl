#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract((p)+0.25)-0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float PI = acos(-1.0);
const float TAU = PI * 2.0;

uniform vec4 param_knob0;

#define p0 paramFetch(param_knob0)

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

float p2f(float p) {
  return exp2((p - 69.0) / 12.0) * 440.0;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  const int N_NOTES = 7;
  const int NOTES[] = int[](0, 5, 7, 12, 17, 19, 24);

  { // notes
    vec2 sum = vec2(0.0);
    repeat(i, 4096) {
      int iGroup = i / 16;
      int iNote = iGroup % N_NOTES;
      float fi = float(i);

      vec3 dice = hash3f(vec3(iNote, iGroup, 2));
      vec3 dice2 = hash3f(vec3(fi, 5, 8));
      vec2 dicen = boxMuller(dice2.xy);

      float l = timeLength.z / floor(exp2(4.0 + dice.x));
      float o = l * dice.y;
      float t = tmod(time - o, l);
      float q = l - t;

      float kdecay = exp2(mix(6.0, 3.0, p0));
      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q) * exp2(-kdecay * t);

      float pitch = 60.0 + TRANSPOSE + float(NOTES[iNote]);
      pitch += 0.04 * dicen.x;
      float freq = p2f(pitch);
      float phase = t * freq + dice2.z;

      float wave = env * sin(TAU * phase);
      sum += vec2(wave) / 64.0 * rotate2D(fi);
    }

    dest += sum;
  }

  return clip(1.3 * tanh(dest));
}
