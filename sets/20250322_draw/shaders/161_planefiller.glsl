#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(i) clamp(i, -1., 1.)
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define lofi(i,m) (floor((i)/(m))*(m))

const float SWING = 0.64;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);
const float MIN3 = pow(2.0, 3.0 / 12.0);
const float P4 = pow(2.0, 5.0 / 12.0);
const float P5 = pow(2.0, 7.0 / 12.0);

// == { CHEATING ZONE START } ======================================================================
// I'm sorry but planefiller is way too difficult to operate without this

// VIBE 0: 1.01 - choir only
// VIBE 1: 1.09 - kick and rim
// VIBE 2: 2.01 - hihat, fm perc, crash, bass
// VIBE 3: 2.09 - hihat+, clap
// VIBE 4: 3.01 - open hihat, snare roll, crash+
// VIBE 5: 4.01 - longer choir, ride, arp
const int VIBE = 0;

const int KICK_FILL = 0x8888;
// const int KICK_FILL = 0x88a6;
// const int KICK_FILL = 0x808f;
// const int KICK_FILL = 0x809e;
// const int KICK_FILL = 0xa18e;

const int SNARE_ROLL_LONGER = 1;

const float TRANSPOSE = 3.0;

// == { CHEATING ZONE END } ========================================================================

uniform vec4 param_knob7; // kick cut

#define p4 paramFetch(param_knob4)
#define p7 paramFetch(param_knob7)

// https://www.shadertoy.com/view/XlXcW4
vec3 hash3f(vec3 s) {
  uvec3 r = floatBitsToUint(s);
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  r = ((r >> 16u) ^ r.yzx) * 1111111111u;
  return vec3(r) / float(-1u);
}

vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D(float x) {
  vec2 v = cis(x);
  return mat2(v.x, v.y, -v.y, v.x);
}

float t2sSwing(float t) {
  float st = 4.0 * t / B2T;
  return 2.0 * floor(st / 2.0) + step(SWING, fract(0.5 * st));
}

float s2tSwing(float st) {
  return 0.5 * B2T * (floor(st / 2.0) + SWING * mod(st, 2.0));
}

vec4 seq16(float t, int seq) {
  t = mod(t, 4.0 * B2T);
  int sti = clamp(int(t2sSwing(t)), 0, 15);
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float i_prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - i_prevStepBehind;
  float prevTime = s2tSwing(prevStep);
  float i_nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + i_nextStepForward;
  float nextTime = s2tSwing(nextStep);

  return vec4(
    prevStep,
    t - prevTime,
    nextStep,
    nextTime - t
  );
}

vec2 shotgun(float t, float spread) {
  vec2 sum = vec2(0.0);

  repeat(i, 64) {
    vec3 dice = hash3f(float(i) + vec3(7, 1, 3));
    sum += vec2(sin(TAU * t * exp2(spread * dice.x))) * rotate2D(TAU * dice.y);
  }

  return sum / 64.0;
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 up = abs(z.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 x = normalize(cross(up, z));
  return mat3(x, cross(z, x), z);
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
  float sidechain = 1.0;

  if (VIBE >= 1) { // kick
    int pattern = 0x8888;
    pattern = time.z < 60.0 * B2T ? pattern : KICK_FILL;

    vec4 seq = seq16(time.y, pattern);
    float t = seq.y;
    float q = seq.w;

    sidechain = smoothstep(0.0, 0.7 * B2T, time.x) * smoothstep(0.0, 0.01 * B2T, B2T - time.x);
    sidechain *= 0.5 + 0.5 * smoothstep(0.0, 0.7 * B2T, t) * smoothstep(0.0, 0.01 * B2T, q);

    {
      float env = smoothstep(0.3, 0.2, t);

      env *= mix(1.0, exp2(-60.0 * t), p7);

      vec2 wave = vec2(0.0);
      vec2 phase = vec2(40.0 * t);
      phase -= 9.0 * exp2(-25.0 * t);
      phase -= 3.0 * exp2(-50.0 * t);
      phase -= 3.0 * exp2(-500.0 * t);
      wave += sin(TAU * phase);

      dest += 0.5 * env * tanh(1.3 * wave);
    }
  }

  if (VIBE >= 2) { // hihat
    const int i_patternCH = VIBE >= 3 ? 0xffff : 0xeaaa;
    vec4 seq = seq16(time.y, i_patternCH);
    float t = seq.y;

    float vel = fract(seq.x * 0.38);
    float env = exp2(-exp2(6.0 - 1.0 * vel - float(mod(seq.x, 4.0) == 2.0)) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0);
    dest += 0.16 * env * mix(0.2, 1.0, sidechain) * tanh(8.0 * wave);
  }

  if (VIBE >= 4) { // open hihat
    vec4 seq = seq16(time.y, 0x2222);
    float t = seq.y;

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.1 * exp2(-14.0 * t) * sidechain * tanh(2.0 * sum);
  }

  if (VIBE >= 2) { // fm perc
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.y;
    float q = seq.w;
    vec3 dice = hash3f(vec3(seq.x, mod(lofi(time.z, 4.0 * B2T), 32.0), 1.0));

    float freq = exp2(9.0 + 2.0 * dice.x);
    float env = exp2(-exp2(3.0 + 5.0 * dice.y) * t) * smoothstep(0.0, 0.01, q);
    float fm = env * exp2(2.0 + 4.0 * dice.z) * sin(freq * exp2(-t));
    float wave = sin(fm);
    dest += 0.05 * sidechain * vec2(wave) * rotate2D(seq.x);
  }

  if (VIBE >= 1) { // rim
    vec4 seq = seq16(time.y, 0x6db7);
    float t = seq.y;

    float env = step(0.0, t) * exp2(-400.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * vec2(wave) * rotate2D(seq.x);
  }

  if (VIBE >= 5) { // ride
    vec4 seq = seq16(time.y, 0xaaaa);
    float t = seq.y;
    float q = seq.w;

    float env = exp2(-4.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 2.9 * env * sin(wave + exp2(13.10 + 0.4 * dice.x) * t + dice2.xy);
      wave = 2.8 * env * sin(wave + exp2(14.97 + 0.4 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(14.09 + 1.0 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.05 * env * mix(0.3, 1.0, sidechain) * tanh(sum);
  }

  if (VIBE >= 3) { // clap
    vec4 seq = seq16(time.y, 0x0808);
    float t = seq.y;
    float q = seq.w;

    float env = mix(
      exp2(-80.0 * t),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-100.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0).xy;

    dest += 0.15 * tanh(20.0 * env * wave);
  }

  if (VIBE >= 4) { // snare909
    float amp = step(60.0 * B2T, time.z) * mix(0.5, 1.0, smoothstep(0.0, 4.0 * B2T, time.y));
    bool roll = 2.0 * B2T <= time.y;

    if (SNARE_ROLL_LONGER != 0) { // longer roll
      amp = smoothstep(32.0 * B2T, 64.0 * B2T, time.z);
      roll = 60.0 * B2T <= time.z;
    }

    vec4 seq = seq16(time.y, 0xffff);
    float t = roll ? mod(time.y, B2T / 6.0) : seq.y;

    float env = exp(-20.0 * t);

    vec2 wave = (
      cyclic(vec3(cis(4000.0 * t), 4000.0 * t), 1.0, 2.0).xy
      + sin(1400.0 * t - 40.0 * exp2(-t * 200.0))
    );

    dest += 0.2 * amp * mix(0.3, 1.0, sidechain) * tanh(4.0 * env * wave);
  }

  if (VIBE >= 2) { // crash
    float t = time.z;
    if (VIBE >= 4) {
      t = mod(t, 60.0 * B2T);
    }

    float env = mix(exp2(-t), exp2(-14.0 * t), 0.7);
    vec2 wave = shotgun(4000.0 * t, 2.5);
    dest += 0.4 * env * mix(0.1, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // chord stuff
    const int N_CHORD = 8;
    const int CHORD[N_CHORD] = int[](
      0, 7, 10, 12, 15, 17, 19, 22
    );

    float t = mod(time.z, 8.0 * B2T);
    float st = max(1.0, lofi(mod(t2sSwing(t) - 1.0, 32.0), 3.0) + 1.0);
    float stt = s2tSwing(st);
    t = mod(t - stt, 8.0 * B2T);
    float nst = min(st + 3.0, 33.0);
    float nstt = s2tSwing(nst);
    float l = nstt - stt;
    float q = l - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    float trans = TRANSPOSE + step(st, 3.0);

    if (VIBE >= 2) { // bass
      float note = 24.0 + trans + float(CHORD[0]);
      float freq = p2f(note);
      float phase = freq * t;
      float wave = tanh(2.0 * sin(TAU * phase));

      dest += 0.5 * sidechain * env * wave;
    }

    env *= mix(
      smoothstep(0.6 * l, 0.4 * l, t - float(VIBE >= 5) * (0.4 * l)),
      exp2(-5.0 * t),
      0.1
    );

    { // choir
      vec2 sum = vec2(0.0);

      repeat(i, 64) {
        float fi = float(i);
        vec3 dice = hash3f(float(i) + vec3(8, 4, 2));

        float note = 48.0 + trans + float(CHORD[i % N_CHORD]);
        float freq = p2f(note) * exp2(0.016 * tan(2.0 * dice.y - 1.0));
        float phase = lofi(t * freq, 1.0 / 16.0);

        vec3 c = vec3(0.0);
        vec3 d = vec3(2.0, -3.0, -8.0);
        float k = 0.5;
        vec2 wave = cyclic(fract(phase) * d, k, 2.0).xy;

        sum += vec2(wave) * rotate2D(fi);
      }

      dest += 0.05 * mix(0.1, 1.0, sidechain) * env * sum;
    }

    if (VIBE >= 5) { // arp
      int iarp = int(16.0 * t / B2T);
      float note = 48.0 + trans + float(CHORD[iarp % N_CHORD]) + 12.0 * float((iarp % 3) / 2);
      float freq = p2f(note);
      float phase = TAU * lofi(t * freq, 1.0 / 16.0);

      vec2 wave = cyclic(vec3(cis(phase), iarp), 0.5, 2.0).xy * rotate2D(time.w);

      dest += 0.2 * sidechain * env * wave;
    }
  }

  return clip(1.3 * tanh(0.9 * dest));
}
