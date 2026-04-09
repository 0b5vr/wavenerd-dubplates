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

mat2 rotate2D(float x) {
  vec2 v = cis(x);
  return mat2(v.x, v.y, -v.y, v.x);
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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  const int N_CHORD = 8;
  int CHORD[] = int[](
    0, 7, 12, 15, 17, 19, 22, 26
  );

  { // key
    vec2 sum = vec2(0.0);
    repeat(i, 8) {
      float t = mod(time.y - 0.1 * B2T * float(i), 4.0 * B2T);
      float q = 4.0 * B2T - t;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp2(-1.0 * t);

      float pitchb = 48.0 + float(CHORD[i]);

      repeat(i, 64) {
        float fi = float(i);
        vec3 dice = hash3f(vec3(i, 8, 2));

        float pitch = pitchb + 0.01 * (dice.x - 0.5);
        float freq = p2f(pitch);
        float phase = t * freq + dice.y;

        vec3 p = vec3(0.5 * cis(TAU * phase) + 10.0, 0.01 * phase);
        sum += cyclic(p, 0.5 * exp2(-2.0 * t), 2.0).xy * rotate2D(TAU * fi / 64.0);
      }

      dest += 1.5 * env * sum / 8.0 / 8.0;
    }
  }

  return clip(1.3 * tanh(dest));
}
