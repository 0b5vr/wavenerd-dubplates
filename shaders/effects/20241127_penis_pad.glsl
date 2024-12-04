// https://bsky.app/profile/0b5vr.bsky.social/post/3l77w6kafoj2f

#define TRANSPOSE -8.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float SWING = 0.6;

const float LN2 = log(2.0);
const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float P5 = pow(2.0, 7.0 / 12.0);

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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  { // chord
    const int CHORD[8] = int[](
      -24, 0, 7, 10, 14, 17, 24, 29
    );

    vec2 sum = vec2(0.0);
    float flutter = 0.001 * cheapnoise(time.w / 64.0).x;
    float fademod = paramFetch(param_knob0);

    repeat(iUnison, 256) {
      float l = 64.0 * B2T;
      float t = mod(time.z - float(iUnison) / 256.0 * l, l);
      float fade = cos(TAU * t / l);
      t += flutter;

      vec3 dice = hash3f(vec3(iUnison, 7, 7));
      vec2 dicen = boxMuller(dice.xy);

      float pitch = 48.0 + TRANSPOSE + float(CHORD[iUnison % 8]);
      float freq = p2f(pitch);

      float phase = t * freq + dice.x;
      phase *= exp(0.01 * dicen.x);

      float radius = 0.1;
      // radius += 0.4 * sin(2.0 * TAU * phase);
      // radius += 0.04 * sin(5.0 * TAU * phase);
      radius *= fademod;

      float pers = 0.5;
      pers *= 0.01 + fademod;

      vec2 wave = cyclic(vec3(radius * cis(TAU * phase), 5.0), pers, 2.0).xy;

      float amp = fade * smoothstep(0.0, freq / p2f(72.0), fademod);
      sum += amp * wave * rotate2D(float(iUnison));
    }

    dest += 0.4 * sum.xx;
  }

  {
    float t = lofi(time.z, 1.0 / 65536.0);
    dest += 0.01 * cheapnoise(128.0 * t);
    dest += 0.03 * fract(t * 60.0);
    dest += 0.01 * fract(t * 60.0 * 320.0);
  }

  return clip(1.3 * tanh(dest));
}
