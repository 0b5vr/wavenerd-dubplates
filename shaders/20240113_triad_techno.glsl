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
      300.0 * t
      - 50.0 * exp(-t * 20.0)
      - 20.0 * exp(-t * 40.0)
      - 10.0 * exp(-t * 400.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // bass
    vec4 seq = seq16(0xffff, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp(-5.0 * t);

    float note = TRANSPOSE + 24.0 + 3.0 * sin(4.0 * seq.s);
    float phase = p2f(note) * t;

    float wave = tanh(5.0 * env * sin(TAU * phase));

    dest += 0.4 * sidechain * wave;
  }

  { // hihat
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.2 + 0.42);
    float env = exp(-exp2(7.0 - 3.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.2 * env * sidechain * tanh(8.0 * wave);
  }

  { // open hihat
    float t = mod(time.x - 2.0 * S2T, B2T);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp(-6.0 * t) * sin(wave + exp2(12.40 + 0.3 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp(-4.0 * t) * sin(wave + exp2(11.57 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp(-9.0 * t) * sin(wave + exp2(15.18 + 0.3 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.14 * exp(-8.0 * t) * sidechain * tanh(sum);
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

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.41 + 0.63);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);
    vec2 wave = cyclicNoise(vec3(cis(200.0 * t), exp2(8.0 + 3.0 * vel) * t), 0.8).xy;
    dest += 0.12 * env * sidechain * tanh(2.0 * wave);
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
      wave = 2.2 * env * sin(wave + exp2(14.67 + 0.5 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(13.89 + 1.0 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.07 * env * sidechain * tanh(sum);
  }

  { // rim
    vec4 seq = seq16(0x6c5e, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;

    float env = exp(-300.0 * t);

    float wave = tanh(4.0 * (
      +tri(t * 400.0 - 0.5 * env)
      +tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * wave * vec2(1, -1);
  }

  { // riff
    const int notes[] = int[](
      0, 0, 5, 12,
      0, 9, 0, -3,
      0, 0, 5, 12,
      0, 9, 0, -3
    );

    vec2 sum = vec2(0.0);

    repeat(i, 4) {
      float fi = float(i);

      vec4 seq = seq16(0x7475, time.y - fi * 3.0 * S2T, 0.25 * B2T);
      float t = seq.t;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, 0.9 * S2T - t);
      env *= mix(1.0, exp(-10.0 * t), 0.5);

      float note = 48.0 + TRANSPOSE + float(notes[int(seq.s)]);
      float freq = p2f(note);
      float phase = freq * t;

      vec2 wave = (
        + sin(TAU * phase + vec2(0.0, PI / 2.0))
        + sin(TAU * MIN3 * phase + vec2(0.0, PI / 2.0) + 1.0)
        + sin(TAU * P5 * phase + vec2(0.0, PI / 2.0) + 2.0)
        + sin(TAU * 2.0 * phase + vec2(0.0, PI / 2.0) + 3.0)
        + sin(TAU * 2.0 * MIN3 * phase + vec2(0.0, PI / 2.0) + 4.0)
        + sin(TAU * 2.0 * P5 * phase + vec2(0.0, PI / 2.0) + 5.0)
      );

      sum += env * wave * exp(-2.0 * fi);
    }

    dest += 0.12 * sum;
  }

  return tanh(dest);
}
