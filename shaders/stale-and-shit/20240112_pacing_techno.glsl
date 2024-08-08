#define TRANSPOSE 4.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define saturate(x) clamp(x, 0., 1.)
#define repeat(i, n) for (int i = 0; i < n; i++)

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

float p2f(float p) {
  return exp2((p - 69.0) / 12.0) * 440.0;
}

float cheapFilterSaw(float phase, float k) {
  float wave = fract(phase);
  float c = smoothstep(1.0, 0.0, wave / k);
  return (wave + c) * 2.0 - 1.0 - k;
}

vec2 cheapFilterSaw(vec2 phase, float k) {
  vec2 wave = fract(phase);
  vec2 c = smoothstep(1.0, 0.0, wave / k);
  return (wave + c) * 2.0 - 1.0 - k;
}

float glidephase(float t, float t1, float p0, float p1) {
  if (p0 == p1) {
    return t * p2f(p1);
  }

  float m0 = (p0 - 69.0) / 12.0;
  float m1 = (p1 - 69.0) / 12.0;
  float b = (m1 - m0) / t1;

  return (
    + p2f(p0) * (pow(2.0, b * min(t, t1)) - 1.0) / b / LN2
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

vec2 cheapnoise(float t){
  uvec3 s=uvec3(t*256.);
  float p=fract(t*256.);

  vec3 dice;
  vec2 v=vec2(0);

  dice=vec3(hash3u(s))/float(-1u)-vec3(.5,.5,0);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));
  dice=vec3(hash3u(s+1u))/float(-1u)-vec3(.5,.5,1);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));
  dice=vec3(hash3u(s+2u))/float(-1u)-vec3(.5,.5,2);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));

  return 2.*v;
}

vec4 seq16( int seq, float t, float tps ) {
  int sti = int( t / tps );
  int rotated = ( ( seq >> ( 15 - sti ) ) | ( seq << ( sti + 1 ) ) ) & 0xffff;

  float prevStepBehind = log2( float( rotated & -rotated ) );
  float prevStep = float( sti ) - prevStepBehind;
  float nextStepForward = 16.0 - floor( log2( float( rotated ) ) );
  float nextStep = float( sti ) + nextStepForward;

  return vec4(
    prevStep,
    t - prevStep * tps,
    nextStep,
    nextStep * tps - t
  );
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float sidechain;

  { // kick
    vec4 seq = seq16(0x8888, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;
    sidechain = 0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(2.0 * B2T, 0.1 * B2T, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    float wave = sin(
      270.0 * t
      - 40.0 * exp(-t * 20.0)
      - 20.0 * exp(-t * 60.0)
      - 10.0 * exp(-t * 300.0)
      - 0.4 * sin(120.0 * t)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // hihat
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.2 + 0.42);
    float env = exp(-exp2(7.0 - 3.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.2 * env * sidechain * tanh(8.0 * wave);
  }

  { // clap
    vec4 seq = seq16(0x0001, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      exp(-26.0 * t),
      exp(-200.0 * mod(t, 0.013)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclicNoise(vec3(4.0 * cis(800.0 * t), 1940.0 * t), 1.5).xy;

    dest += 0.15 * tanh(20.0 * env * wave);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.41 + 0.63);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);
    vec2 wave = cyclicNoise(vec3(cis(200.0 * t), exp2(8.0 + 3.0 * vel) * t), 0.8).xy;
    dest += 0.15 * env * sidechain * tanh(2.0 * wave);
  }

  { // perc 1
    float t = mod(time.z - S2T, 6.0 * S2T);

    float env = mix(
      exp(-t),
      exp(-30.0 * t),
      0.95
    );
    vec2 wave = sin(7100.0 * t + vec2(0, PI / 2.0) + 10.0 * cheapnoise(t));
    dest += 0.3 * env * tanh(wave);
  }

  { // perc 2
    float t = mod(time.z - 3.0 * S2T, 6.0 * S2T);

    float env = mix(
      exp(-t),
      exp(-30.0 * t),
      0.95
    );
    vec2 wave = 2.0 * fract(1200.0 * t + sin(1000.0 * t) + vec2(0.0, 0.25)) - 1.0;
    dest += 0.3 * env * tanh(wave);
  }

  { // beep
    float t = mod(time.y - 5.0 * S2T, 16.0 * S2T);

    float env = smoothstep(0.0, 0.001, t) * mix(
      exp(-t),
      smoothstep(0.0, 0.001, 0.07 - t),
      0.98
    );
    vec2 wave = sin(50000.0 * t + vec2(PI / 2.0, 0));
    dest += 0.2 * env * wave;
  }

  { // oidos drone
    vec2 sum=vec2(0.0);

    repeat(i, 2500) {
      vec3 diceA = hash3f(vec3(i / 50));
      vec3 diceB = hash3f(vec3(i));

      float t = mod(time.z - diceA.x * (64.0 * B2T), 64.0 * B2T);
      float env = sin(PI * t / (64.0 * B2T));

      float tone = mix(8.0, 17.0, diceA.y) + 0.06 * diceB.y;
      float freq = exp2(tone);
      vec2 phase = t * freq + fract(diceB.xy * 999.0);
      phase += 0.1 * fract(32.0 * phase); // add high freq

      sum += sin(TAU * phase) * env / 1000.0;
    }

    dest += 1.0 * mix(0.2, 1.0, sidechain) * sum;
  }


  return tanh(1.2 * dest);
}
