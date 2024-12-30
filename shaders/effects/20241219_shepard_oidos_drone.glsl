#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  { // oidos drone
    vec2 sum=vec2(0.0);

    repeat(i, 2500) {
      vec3 diceA = hash3f(vec3(i / 50) + vec3(28, 18, 26));
      vec3 diceB = hash3f(vec3(i));

      float l = 64.0 * B2T;
      float t = mod(time.z - diceA.x * l, l);
      float env = sin(PI * t / l);

      float tone0 = 7.0 + 3.0 * diceA.y + 0.05 * diceB.y;
      float tone1 = 1.0 + tone0;
      float b = (tone1 - tone0) / l;

      vec2 phase = vec2(0.0);
      // \int_0^t 2^{a + bt} = \frac{2^a (2^{bt} - 1)}{b \log 2}
      phase += exp2(tone0) * (exp2(b * t) - 1.0) / (b * LN2);
      phase += fract(diceB.xy * 999.0); // random phase offset
      phase += 0.1 * fract(16.0 * phase); // add high freq

      sum += sin(TAU * phase) * env / 1000.0;
    }

    dest += sum;
  }

  return clip(1.3 * tanh(dest));
}
