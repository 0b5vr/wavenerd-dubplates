#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(i) clamp(i, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
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

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float duck = 1.0;

  { // grain
    float t = time.y;
    float q = 4.0 * B2T - t;

    float env = smoothstep(0.0, 0.01, q);
    env *= mix(
      exp(-20.0 * t),
      exp(-t),
      0.002
    );

    const int GRAIN_COUNT = 4;
    const float GRAIN_LENGTH = 0.02;
    const float GRAIN_INTERVAL = GRAIN_LENGTH / float(GRAIN_COUNT);

    vec2 sum = vec2(0.0);
    repeat(i, GRAIN_COUNT) {
      float fi = float(i);

      float off = -GRAIN_INTERVAL * fi;
      float tg = mod(t + off, GRAIN_LENGTH);
      float prog = tg / GRAIN_LENGTH;

      vec3 dice = hash3f(vec3(i, lofi(t + off, GRAIN_LENGTH), 10));
      vec2 dicen = boxMuller(dice.xy);

      float envg = smoothstep(0.0, 0.5, prog) * smoothstep(1.0, 0.5, prog);

      float freq = 2000.0;
      vec2 phase = vec2(freq * t);
      float spread = 1.0;
      phase *= exp2(spread * dicen.xy);
      phase += dice.xy;

      vec2 wave = sin(TAU * phase);
      sum += 2.0 * envg * wave / float(GRAIN_COUNT);
    }

    dest += 0.2 * env * sum;
  }

  return clip(1.3 * tanh(dest));
}
