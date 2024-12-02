#define TRANSPOSE 7.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define saturate(x) clamp(x, 0., 1.)
#define repeat(i, n) for (int i = 0; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p)-0.5))

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

vec2 ladderLPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 4.0 * omega * (omegaSq - 1.0);
  float b = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;

  return vec2(
    1.0 / sqrt(a * a + b * b),
    atan(a, b)
  );
}

vec2 twoPoleHPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 2.0 * (1.0 - reso) * omega;
  float b = omegaSq - 1.0;

  return vec2(
    omegaSq / sqrt(a * a + b * b),
    atan(a, b)
  );
}

vec4 seq16( int seq, float t, float tps ) {
  int sti = int( t / tps ) & 15;
  int rotated = ( ( seq >> ( 15 - sti ) ) | ( seq << ( sti + 1 ) ) ) & 0xffff;

  float prevStepBehind = log2( float( rotated & -rotated ) );
  float prevStep = float( sti ) - prevStepBehind;
  float nextStepForward = 16.0 - floor( log2( float( rotated ) ) );
  float nextStep = float( sti ) + nextStepForward;

  return vec4(
    prevStep,
    mod( t, 16.0 * tps ) - prevStep * tps,
    nextStep,
    nextStep * tps - mod( t, 16.0 * tps )
  );
}

vec4 quant(float x, float ks, float kt, out float i) {
  i = floor(floor(x / ks + 1E-4) * ks / kt + 1E-4);

  float s = kt <= ks
    ? ks * floor(x / ks + 1E-4)
    : ks * ceil(i * kt / ks - 1E-4);
  float l = kt <= ks
    ? ks
    : ks * ceil((i + 1.0) * kt / ks - 1E-4) - s;

  float t = x - s;
  float q = l - t;

  return vec4(s, t, s + l, q);
}

vec4 quant(float x, float ks, float kt) {
  float i;
  return quant(x, ks, kt, i);
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float sidechain;

  { // kick
    vec4 seq = seq16(0x8888, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;
    sidechain = 0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.5, 0.3, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    float tt = t + 0.001 * sin(130.0 * t);
    float wave = sin(
      290.0 * tt
      - 40.0 * exp(-tt * 20.0)
      - 30.0 * exp(-tt * 40.0)
      - 10.0 * exp(-tt * 200.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // lownoise
    float t = mod(time.y, 4.0 * S2T);

    vec2 wave = cyclicNoise(vec3(cis(300.0 * t), 50.0 * t), 3.0).xy;
    wave += sin(300.0 * t);
    dest += 0.5 * sidechain * tanh(wave);
  }

  { // hihat
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.38);
    float env = exp(-exp2(5.0 - 1.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.12 * env * sidechain * tanh(8.0 * wave);
  }

  { // hihat long
    float t = mod(time.x - 2.0 * S2T, B2T);

    float env = exp(-20.0 * t);
    vec2 wave = shotgun(3000.0 * t, 1.2, 0.0, 1.3);
    dest += 0.2 * env * sidechain * tanh(8.0 * wave);
  }

  { // ride
    float t = mod(time.x, 2.0 * S2T);

    float env = exp(-5.0 * t);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * env * sin(wave + exp2(11.10 + 0.1 * dice.x) * t + dice2.xy);
      wave = 2.2 * env * sin(wave + exp2(14.67 + 0.5 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(13.89 + 1.0 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.05 * env * sidechain * tanh(sum);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = 0.5 + 0.5 * cos(PI / 2.0 * time.x / S2T + 1.0);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 2.0 * vel) * t);
    vec2 wave = cyclicNoise(vec3(cis(200.0 * t), exp2(9.0 + 2.0 * vel) * t), 0.8).xy;
    dest += 0.15 * env * sidechain * tanh(2.0 * wave);
  }

  { // clap
    vec4 seq = min(
      quant(mod(time.z - 2.0 * S2T, 64.0 * S2T), S2T, 5.28 * S2T),
      quant(mod(time.z - 2.0 * S2T, 64.0 * S2T), S2T, 7.61 * S2T)
    );
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      exp(-26.0 * t),
      exp(-200.0 * mod(t, 0.013)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclicNoise(vec3(4.0 * cis(900.0 * t), 1940.0 * t), 1.5).xy;

    dest += 0.2 * mix(0.3, 1.0, sidechain) * tanh(8.0 * env * wave);
  }

  { // crash
    float t = time.z;

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 3.0);
    dest += 0.3 * env * sidechain * tanh(8.0 * wave);
  }

  { // additive riff
    vec2 sum = vec2(0.0);

    float t = mod(time.x, S2T);
    float q = S2T - t;
    float st = floor(time.z / S2T);
    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    float basefreq = 80.0;
    float stmod = fract(0.615 * st);

    float cutenv = smoothstep(0.0, 0.01, t) * exp(-10.0 * t);
    float cutoff = exp2(7.0 + 3.0 * stmod + 4.0 * cutenv);

    repeat(i, 128) {
      float fi = float(i);

      float p = 1.0 + fi;
      p = pow(p, 1.1); // metal
      p = mix(p, 1.0, 0.1); // centroid
      float freq = basefreq * p;

      vec2 phase = vec2(t * freq);
      vec2 lpf = ladderLPF(freq, cutoff, 0.3);
      vec2 hpf = twoPoleHPF(freq, 200.0, 0.0);

      sum += sin(TAU * phase + lpf.y + hpf.y) / p * lpf.x * hpf.x * env * rotate2D(2.4 * fi);
    }

    dest += 0.3 * mix(0.3, 1.0, sidechain) * tanh(5.0 * sum);
  }

  return tanh(dest);
}
