#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(i) clamp(i, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
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

float cheapfiltersaw(float phase, float k) {
  float wave = mod(phase, 1.0);
  float c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c) * 2.0 - 1.0 - k;
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

vec4 quant(float x, float ks, float kt) {
  float i = floor(floor(x / ks + 1E-4) * ks / kt + 1E-4);
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

    float tt = t;
    float wave = sin(
      250.0 * tt
      - 30.0 * exp(-tt * 10.0)
      - 20.0 * exp(-tt * 30.0)
      - 20.0 * exp(-tt * 200.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // hihat
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.38);
    float env = exp(-exp2(5.0 - 1.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.2 * env * sidechain * tanh(8.0 * wave);
  }

  { // snare
    float t = mod(time.y - B2T, 2.0 * B2T);
    float q = 2.0 * B2T - t;

    float env = exp(-10.0 * t) * smoothstep(0.0, 0.001, q);

    float phase = 200.0 * t - 10.0 * exp(-400.0 * t);

    vec2 wave = vec2(0.0);
    wave += exp(-20.0 * t) * (
      + cis(TAU * phase)
      + cis(1.5 * TAU * phase)
    );
    wave += exp(-10.0 * t) * cheapnoise(50.0 * t);

    dest += 0.25 * env * mix(0.3, 1.0, sidechain) * tanh(8.0 * wave);
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

  { // crash
    float t = time.z;

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 3.0);
    dest += 0.3 * env * sidechain * tanh(8.0 * wave);
  }

  { // shepard bass
    vec2 sum = vec2(0.0);

    float t = mod(time.x, S2T);
    float q = S2T - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q - 0.01);

    repeat(i, 64) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi));
      float shepardPhase = fract(lofi(time.z, B2T) / (256.0 * S2T) - mod(fi, 4.0) / 4.0);

      float pitch = 36.0 + TRANSPOSE;
      pitch += 48.0 * (shepardPhase - 0.5);
      float freq = p2f(pitch);
      float phase = t * freq;
      float phaset = phase * exp(0.04 * (dice.x - 0.5)) + dice.y;

      vec2 wave = vec2(sin(TAU * phase));
      wave += 2.0 * vec2(cheapfiltersaw(2.0 * phaset, exp(-1.0 * t))) * rotate2D(fi);

      sum += wave / 16.0 * env * sin(PI * shepardPhase);
    }

    dest += 0.4 * sidechain * tanh(sum);
  }

  { // shepard arp
    const int N_ARPNOTES = 7;
    const int ARPNOTES[N_ARPNOTES] = int[](
      0, 14, 3, 7, 10, 15, 22
    );

    vec2 sum = vec2(0.0);

    repeat(iDelay, 3) {
      float fiDelay = float(iDelay);
      float off = -3.0 * S2T * fiDelay;

      const float ARP_DIVISION = 4.0;
      float t = mod(time.x + off, B2T / ARP_DIVISION);
      float st = mod(floor((time.z + off) / (B2T / ARP_DIVISION)), 256.0);

      float q = B2T / ARP_DIVISION - t;
      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);

      repeat(i, 64) {
        float fi = float(i);
        vec3 dice = hash3f(vec3(fi));
        float shepardPhase = fract(lofi(time.z, B2T) / (256.0 * S2T) - mod(fi, 4.0) / 4.0);

        int inote = int(mod(st, float(N_ARPNOTES)));
        float pitch = 48.0 + TRANSPOSE + float(ARPNOTES[inote]) + 0.04 * (dice.y - 0.5);
        pitch += 48.0 * (shepardPhase - 0.5);
        float freq = p2f(pitch);
        float phase = t * freq + dice.z;

        vec2 wave = vec2(0.0);
        wave += cheapfiltersaw(phase, exp(-1.0 * t));

        sum += wave / 16.0 * env * exp(-fiDelay) * sin(PI * shepardPhase) * rotate2D(fi);
      }
    }

    dest += 0.6 * mix(0.2, 1.0, sidechain) * sum;
  }

  { // shepard pad
    const int chord[7] = int[](
      0, 3, 7, 10, 14, 19, 22
    );

    vec2 sum = vec2(0.0);

    float t = time.x;

    repeat(i, 128){
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi));
      float shepardPhase = fract(lofi(time.z, B2T) / (256.0 * S2T) - mod(fi, 4.0) / 4.0);

      int inote = int(7.0 * dice.x);
      float pitch = 48.0 + TRANSPOSE + float(chord[inote]) + 0.6 * (dice.y - 0.5);
      pitch += 48.0 * (shepardPhase - 0.5);
      float freq = p2f(pitch);
      float phase = t * freq + dice.z;

      float z = 15.0;
      vec3 p1 = vec3(2.0 * cis(TAU * phase), z);
      vec3 p2 = vec3(2.0 * cis(TAU * phase + 0.4), z);
      vec2 wave = cyclicNoise(p1, 2.0).xy - cyclicNoise(p2, 2.0).xy;

      sum += wave / 24.0 * sin(PI * shepardPhase) * rotate2D(fi);
    }

    dest += 0.2 * mix(0.2, 1.0, sidechain) * tanh(sum);
  }

  return clip(1.3 * tanh(dest));
}
