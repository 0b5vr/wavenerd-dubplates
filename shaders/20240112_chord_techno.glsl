#define TRANSPOSE 5.0

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
  if (p0 == p1 || t1 == 0.0) {
    return t * p2f(p1);
  }

  float m0 = (p0 - 69.0) / 12.0;
  float m1 = (p1 - 69.0) / 12.0;
  float b = (m1 - m0) / t1;

  return (
    + p2f(p0) * (
      + min(t, 0.0)
      + (pow(2.0, b * clamp(t, 0.0, t1)) - 1.0) / b / LN2
    )
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

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    float wave = sin(
      240.0 * t
      - 50.0 * exp(-t * 20.0)
      - 10.0 * exp(-t * 50.0)
      - 10.0 * exp(-t * 200.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // bass
    vec4 seq = seq16(0x6d6d, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp(-5.0 * t);

    float note = TRANSPOSE + 24.0;
    float phase = p2f(note) * t;

    float wave = tanh(3.0 * sin(
      TAU * phase
      + 0.8 * env * sin(200.0 * TAU * exp(-3.0 * t))
    ));

    dest += 0.4 * env * sidechain * wave;
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
    vec4 seq = seq16(0x0808, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      exp(-30.0 * t),
      exp(-200.0 * mod(t, 0.013)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclicNoise(vec3(4.0 * cis(800.0 * t), 1940.0 * t), 1.5).xy;

    dest += 0.14 * tanh(20.0 * env * wave);
  }

  { // ride
    float t = mod(time.x, 2.0 * S2T);

    float env = exp(-5.0 * t);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * env * sin(wave + exp2(12.10 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * env * sin(wave + exp2(14.87 + 0.1 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(13.89 + 0.1 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.07 * env * sidechain * tanh(sum);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.41 + 0.63);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);
    vec2 wave = cyclicNoise(vec3(cis(200.0 * t), exp2(8.0 + 3.0 * vel) * t), 0.8).xy;
    dest += 0.15 * env * sidechain * tanh(2.0 * wave);
  }

  { // pad
    vec2 sum=vec2(0);

    const int pitchTable[8] = int[](0, 3, 7, 10, 12, 14, 17, 19);

    repeat(i, 64) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi));

      float t = mod(time.z, 8.0 * B2T);

      float note = TRANSPOSE + 48.0 + float(pitchTable[i % 8]);
      float detune = exp2(0.008 * boxMuller(dice.xy).x);
      float phase = glidephase(t - 0.9 * B2T, 0.5 * B2T, note + 6.0, note) * detune + dice.z;

      vec3 p = vec3(cis(TAU * phase), t) + 8.5;
      vec2 wave = cyclicNoise(p, 1.7).xy - cyclicNoise(p + 0.4, 1.7).xy;

      sum += wave * rotate2D(fi) / 24.0;
    }

    dest += 0.9 * mix(0.2, 1.0, sidechain) * tanh(sum);
  }

  return tanh(dest);
}
