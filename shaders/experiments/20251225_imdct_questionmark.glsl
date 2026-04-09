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

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
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
  float duck = 1.0;

  { // imdct?
    float l = S2T;
    float t = mod(time.x, l);
    float q = l - t;

    float st = floor(time.z / l);
    vec3 dice = hash3f(vec3(st, 10, 10));

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
    env *= exp2(-exp2(mix(4.0, 5.0, dice.x)) * t);

    const float RATE = 48000.0;
    const int SIZE = 512;
    const int BINS = 512;
    int totalSample = int(t * RATE);
    vec2 sum = vec2(0.0);

    repeat(i, 2) {
      int frameSample = (totalSample + i * SIZE) % (SIZE + SIZE);
      int frameHead = totalSample - frameSample;
      float frameHeadSec = float(frameHead) / RATE;

      float x = float(frameSample) / float(SIZE);
      float window = sin(PI / 2.0 * sin(PI / 2.0 * x) * sin(PI / 2.0 * x)); // vorbis window

      repeat(i, BINS) {
        float fi = float(i);
        float bin = (fi + 0.5) / float(SIZE);
        float logbin = log2(bin);

        vec2 amp = vec2(1.0);
        amp = mix(
          amp,
          vec2(pow(0.5 + 0.5 * sin(TAU * log2(1.618) * logbin), 4.0)),
          0.5
        ); // comb
        amp = mix(
          amp,
          vec2(exp2(-pow((logbin + 4.0) / 1.0, 2.0))),
          0.5
        ); // peaking
        // amp *= smoothstep(0.0, -8.0, logbin); // slope
        amp = pow(vec2(20.0), amp - 1.0); // linear -> log
        amp *= exp2(-2.0 + 3.0 * cyclic(vec3(
          logbin,
          exp(-frameHeadSec * 10.0),
          100.0 * dice.y
        ), 0.5, 2.0).x); // random macro band
        amp *= cyclic(vec3(
          5.0 * logbin,
          500.0 * frameHeadSec,
          100.0 * dice.z
        ), 0.5, 2.0).xy; // random micro bin
        // amp = mix(amp, amp.yx, 0.3); // stereo merge
        amp *= smoothstep(-8.0, -7.0, logbin); // low cut
        // amp *= smoothstep(-1.0, -1.2, logbin); // high cut

        sum += amp * window * cos(PI * x * (fi + 0.5));
      }
    }

    // dest += 0.5 * sum;
    dest += 0.5 * env * tanh(sum);
  }

  return clip(1.3 * tanh(dest));
}
