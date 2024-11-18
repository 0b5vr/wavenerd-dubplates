#define TRANSPOSE 6.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define repeat(i, n) for (int i = ZERO; i < n; i++)
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

vec2 ladderLPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;
  float b = 4.0 * omega * (omegaSq - 1.0);

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

  return vec4(s / kt, t, (s + l) / kt, q);
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

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    vec2 wave = sin(
      260.0 * t
      - 60.0 * exp(-t * 10.0)
      - 30.0 * exp(-t * 40.0)
      - 20.0 * exp(-t * 400.0)
      - 0.5 * cheapnoise(2.0 * t) * exp(-t * 10.0)
    );
    dest += 0.6 * tanh(3.0 * env * wave);
  }

  { // sub kick
    float t = mod(time.x - S2T, S2T);
    float q = S2T - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q) * exp(-10.0 * t);
    float wave = sin(300.0 * t - 4.0 * exp(-40.0 * t));
    dest += 0.5 * sidechain * env * wave;
  }

  { // low freq noise
    float t = time.x;

    float phase = 300.0 * t;
    vec2 wave = cyclicNoise(vec3(cis(phase), 414.0 * t), 4.0).xy;
    wave += 0.5 * sin(phase);
    dest += 0.3 * sidechain * tanh(2.0 * wave);
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
      wave = 4.5 * exp(-6.0 * t) * sin(wave + exp2(12.80 + 0.3 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp(-4.0 * t) * sin(wave + exp2(10.88 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp(-9.0 * t) * sin(wave + exp2(14.92 + 0.3 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.14 * exp(-5.0 * t) * sidechain * tanh(2.0 * sum);
  }

  { // clap
    vec4 seq = seq16(0x0808, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      exp(-10.0 * t),
      exp(-100.0 * mod(t, 0.014)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclicNoise(vec3(4.0 * cis(600.0 * t), 2940.0 * t), 1.3).xy;

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
    const int N_NOTES = 6;
    const int NOTES[] = int[](0, 6, 12, 0, 5, 8);

    const int N_CHORD = 3;
    const int CHORD[N_CHORD] = int[](0, 6, 7);

    float repeatTime = 9.0 * S2T;

    vec2 sum = vec2(0.0);

    repeat(i, 4) {
      float fi = float(i);
      float delayTime = 3.0 * fi * S2T;
      float delayDecay = exp(-1.0 * fi);

      float seqi;
      vec4 seq = quant(mod(time.z - delayTime, repeatTime), S2T, 1.24 * S2T, seqi);
      float t = seq.t;
      float q = seq.q;
      seqi = mod(round(seqi), float(N_NOTES));

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp(-10.0 * t);

      float cutoff = exp2(7.0 + 7.0 * env - 0.7 * fi);

      vec2 wave = vec2(0.0);
      repeat(j, N_CHORD) {
        float fj = float(j);

        float note = 48.0 + TRANSPOSE + float(CHORD[j]) + float(NOTES[int(seqi)]);
        float freq = p2f(note);

        repeat(k, 128) {
          float fk = float(k);
          float p = 1.0 + fk;
          float freqp = freq * p;

          vec2 lpf = ladderLPF(freqp, cutoff, 0.2);
          vec2 hpf = twoPoleHPF(freqp, 100.0, 0.0);
          float phase = TAU * freqp * t + lpf.y + hpf.y;

          wave += vec2(sin(phase)) / p * lpf.x * hpf.x * rotate2D(2.4 * fk + fi);
        }
      }

      sum += env * delayDecay * tanh(wave);
    }

    dest += 0.4 * sum;
  }

  return tanh(dest);
}
