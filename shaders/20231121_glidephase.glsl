#define TRANSPOSE 4.0

#define ZERO min(0, int(bpm))
#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define saturate(x) clamp(x, 0., 1.)
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float LN2 = log(2.0);
const float TAU = acos(-1.0) * 2.0;

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
    vec4 seq = seq16(0x8200, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;
    sidechain = 0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    float wave = sin(
      270.0 * t
      - 30.0 * exp(-t * 20.0)
      - 20.0 * exp(-t * 60.0)
      - 10.0 * exp(-t * 500.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // bass
    const int steps[] = int[](0, 6, 26, 32, 54, 58, 64);
    const int notes[] = int[](-2, 0, 12, 15, 8, 7, -2);

    int st = int(time.z / S2T) % 64;
    int i = 0;
    for (i = 0; i < 7; i++) {
      if (st < steps[i]) {
        break;
      }
    }

    float t = mod(time.z, 64.0 * S2T) - float(steps[i - 1]) * S2T;
    float l = float(steps[i] - steps[i - 1]) * S2T;
    float q = l - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
    float fmenv = exp(-8.0 * t);

    float p0 = 24.0 + TRANSPOSE + float(notes[i - 1]);
    float p1 = 24.0 + TRANSPOSE + float(notes[i]);

    // sub bass
    float phase = glidephase(t, 2.0 * S2T, p0, p1);
    float wave = sin(TAU * phase);
    dest += 0.2 * sidechain * tanh(2.0 * env * wave);

    // unison
    repeat(iUnison, 4) {
      vec3 dice = hash3f(vec3(iUnison, 4, 4));
      vec2 dicen = boxMuller(dice.xy);

      vec2 phaseu = phase * exp2(0.03 * dicen.xy) + dice.xy;
      vec2 wave = cheapFilterSaw(phaseu, 0.04);
      dest += 0.1 * sidechain * tanh(2.0 * env * wave);
    }
  }

  { // hihat
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.2 + 0.62);
    float env = exp(-exp2(6.0 - 3.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.2 * env * sidechain * tanh(8.0 * wave);
  }

  { // clap
    vec4 seq = seq16(0x0820, time.y, 0.25 * B2T);
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      exp(-26.0 * t),
      exp(-200.0 * mod(t, 0.013)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclicNoise(vec3(4.0 * cis(900.0 * t), 1940.0 * t), 1.5).xy;

    dest += 0.15 * tanh(20.0 * env * wave);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.41 + 0.62);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);
    vec2 wave = cyclicNoise(vec3(cis(600.0 * t), exp2(10.0 + 3.0 * vel) * t), 0.8).xy;
    dest += 0.15 * env * sidechain * tanh(2.0 * wave);
  }

  { // crash
    float t = time.z;

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 3.0);
    dest += 0.3 * env * sidechain * tanh(8.0 * wave);
  }

  { // glide lead
    const int N_NOTES = 16;
    int notes[N_NOTES] = int[](
      0, 7, 14, 7,
      15, 22, 7, 26,
      0, 22, 17, 19,
      7, 14, 15, 7
    );

    repeat(iStep, N_NOTES) {
      float l = S2T;
      float t = mod(time.y + S2T * float(N_NOTES - iStep), 16.0 * S2T);

      float p0 = 60.0 + TRANSPOSE + float(notes[max(0, iStep - 1) % N_NOTES]);
      float p1 = 60.0 + TRANSPOSE + float(notes[iStep]);
      float phase = glidephase(t, 0.02, p0, p1);

      float delayDecay = exp(-5.0 * t);
      t = mod(t, 3.0 * S2T);
      float q = l - t;

      float env = smoothstep(0.0, 0.001, t) * mix(0.2, 1.0, smoothstep(0.0, 0.01, q));
      float filt = 1.0 - 1.0 * exp(-2.0 * t);

      vec2 sum = vec2(0.0);

      repeat(iUnison, 4) {
        vec3 dice = hash3f(vec3(iUnison, 1, 12));
        vec2 dicen = boxMuller(dice.xy);

        vec2 phaseu = phase * exp2(0.01 * dicen.xy - float(iUnison % 2)) + dice.xy;
        vec2 wave = cheapFilterSaw(phaseu, filt);
        sum += 0.1 * sidechain * delayDecay * env * wave;
      }

      dest += sum * rotate2D(float(iStep));
    }
  }

  { // chord
    int chords[16] = int[](
      0, 3, 7, 10, 17, 19, 24, 26, // Im
      -4, 0, 3, 7, 10, 12, 17, 22  // VIb
    );

    float t = mod(time.z, 32.0 * S2T);
    float q = 32.0 * S2T - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

    vec2 sum = vec2(0.0);

    vec4 seq = seq16(0x9627, time.y, S2T);
    float curve = seq.x >= 14.0
      ? exp(-6.0 * mod(seq.t, 0.5 * S2T))
      : exp(-2.0 * seq.t);
    float radius = mix(0.1, 0.4, curve);
    float pers = exp2(mix(2.0, -1.0, curve));

    repeat(iUnison, 64) {
      vec3 dice = hash3f(vec3(iUnison, 7, 7));
      vec2 dicen = boxMuller(dice.xy);

      int prog = 8 * int(time.z / 32.0 / S2T);
      float p0 = 48.0 + TRANSPOSE + float(chords[iUnison % 8 + (prog + 8) % 16]);
      float p1 = 48.0 + TRANSPOSE + float(chords[iUnison % 8 + prog % 16]);

      float phase = glidephase(t, 8.0 * S2T, p0, p1) + dice.x;
      phase *= exp(0.01 * dicen.x);

      vec2 wave = cyclicNoise(vec3(radius * cis(TAU * phase), 8.0), pers).xy;
      sum += 0.04 * env * wave * rotate2D(float(iUnison));
    }

    dest += sum;
  }

  return tanh(dest);
}
