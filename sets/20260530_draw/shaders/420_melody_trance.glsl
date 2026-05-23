// - introduce 16th hihat
// - introduce clap + shaker
// - sweep (p1)
// - mute rhythms + bass, enable chord progression + chord
// - gradually open chord cutoff (p4)
// - unmute kick + bass, introduce lead (p0)
// - unmute hihat
// - snare roll! (p2)
// - prepare for full unmute
// - WOO YEAH
// - remove clap + open hihat, remove chord progression
// - unmute rhythms other than kick + bass + hihat

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define u2b(u) ((u) * 2.0 - 1.0)
#define b2u(b) ((b) * 0.5 + 0.5)
#define tri(x) (1.0 - 4.0 * abs(fract((x) + 0.25) - 0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define TRANSPOSE 0.0

const float SWING = 0.5;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

uniform vec4 param_knob0; // lead vol
uniform vec4 param_knob1; // sweep
uniform vec4 param_knob2; // snare roll
uniform vec4 param_knob3; // kick cut
uniform vec4 param_knob4; // chord cut

#define p0 paramFetch(param_knob0)
#define p1 paramFetch(param_knob1)
#define p2 paramFetch(param_knob2)
#define p3 paramFetch(param_knob3)
#define p4 paramFetch(param_knob4)

int imod(int x, int y) {
  return ((x % y) + y) % y;
}

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

mat2 rotate2D(float t) {
  float c = cos(t);
  float s = sin(t);
  return mat2(c, s, -s, c);
}

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
}

float tmod(vec4 time, float d) {
  vec4 t = mod(time, timeLength);
  float offset = lofi(t.z - t.x + timeLength.x / 2.0, timeLength.x);
  offset -= lofi(t.z, d);
  return t.x + offset;
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

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

float glidephase(float t, float t1, float pitch0, float pitch1) {
  if (pitch0 == pitch1) {
    return t * p2f(pitch1);
  }

  float m0 = (pitch0 - 69.0) / 12.0;
  float m1 = (pitch1 - 69.0) / 12.0;
  float b = (m1 - m0) / t1;

  return (
    + p2f(pitch0) * (pow(2.0, b * min(t, t1)) - 1.0) / b / LN2
    + max(0.0, t - t1) * p2f(pitch1)
  );
}

vec3 cyclic(vec3 p, float pers, float lacu) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  repeat(i, 5) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum /= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
}

float cheapfiltersaw(float phase, float k) {
  float wave = fract(phase);
  float c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
}

vec2 cheapfiltersaw(vec2 phase, float k) {
  vec2 wave = fract(phase);
  vec2 c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
}

float cheapfiltersquare(float phase, float k) {
  float s = floor(2.0 * fract(phase));
  float c = smoothstep(1.0, 0.0, fract(2.0 * phase) / (1.0 - k));
  return 2.0 * mix(c, 1.0 - c, s) - 1.0;
}

vec2 cheapfiltersquare(vec2 phase, float k) {
  vec2 s = floor(2.0 * fract(phase));
  vec2 c = smoothstep(1.0, 0.0, fract(2.0 * phase) / (1.0 - k));
  return 2.0 * mix(c, 1.0 - c, s) - 1.0;
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

float getChordNote(int i, float t) {
  const int N_CHORD_NOTES = 5;
  const int CHORDS[] = int[](
    -12+5, 0, 4, 7, 12,
    -12+5, 0, 4, 7, 11,
    -12+4, 2, 4, 7, 11,
    -12+4, 2, 4, 7, 12,

    -12+2, 2, 4, 7, 12,
    -12+8, 2, 4, 7, 11,
    -12+9, 2, 4, 7, 11,
    -12+7, -12+10, 4, 7, 12,

    -12+5, 0, 4, 7, 12,
    -12+5, 0, 4, 7, 11,
    -12+4, 0, 2, 7, 11,
    -12+9, 2, 4, 7, 11,

    -12+2, -12+9, 5, 7, 12,
    -12+4, -12+11, 2, 7, 12,
    -12+5, 0, 4, 7, 12,
    -12+7, 0, 2, 7, 12
  );

  int iProg = 0;
  // { // enable progression
  //   iProg = imod(int(t / (4.0 * B2T)), 16);
  // }

  int j = (i % N_CHORD_NOTES) + N_CHORD_NOTES * iProg;
  return TRANSPOSE + float(CHORDS[j]);
}

vec2 mainAudioDry(vec4 time) {
  vec2 dest = vec2(0.0);

  float duck = smoothstep(0.0, 0.4, time.x) * smoothstep(0.0, 0.001, B2T - time.x);

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;

    if (time.z > 60.0 * B2T) { // 16 bar break
      t = time.y;
      q = 4.0 * B2T - t;
    }

    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    float env = smoothstep(0.0, 0.001, q);
    env *= smoothstep(0.3, 0.1, t);
    env *= mix(1.0, exp2(-80.0 * t), p3); // hpf-like

    vec2 phase = vec2(
      40.0 * t
      - 10.0 * exp2(-t * 30.0)
      - 6.0 * exp2(-t * 100.0)
      - 6.0 * exp2(-t * 600.0)
    );

    vec2 wave = tanh(2.0 * sin(TAU * phase));

    dest += 0.6 * env * wave;
  }

  { // bass
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s;
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    float pitch = 36.0 + getChordNote(0, time.z);
    pitch += mod(st, 4.0) == 2.0 ? 12.0 : 0.0;
    float freq = p2f(pitch);
    vec2 phase = vec2(freq * t);

    vec2 wave = sin(TAU * phase);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i, 30, 18));
      vec2 phaseu = phase * exp2(0.02 * (dice.x - 0.5)) + dice.yz;

      float k = exp2(-2.0 * t);
      wave += 0.5 * cheapfiltersquare(phaseu, k);
    }

    // reduce side
    wave = mix(vec2(dot(wave, vec2(0.5))), wave, 0.2);

    dest += 0.3 * mix(0.0, 1.0, duck) * env * wave;
  }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-30.0 * t);

  //   vec2 wave = shotgun(2800.0 * t, 2.6, 0.4, 1.0);
  //   wave = tanh(6.0 * wave);

  //   dest += 0.3 * mix(0.1, 1.0, duck) * env * wave;
  // }

  // { // clap
  //   vec4 seq = seq16(time.y, 0x0808);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = mix(
  //     exp2(-20.0 * t),
  //     exp2(-500.0 * mod(t, 0.012)),
  //     exp2(-100.0 * max(0.0, t - 0.02))
  //   );

  //   vec2 wave = cyclic(vec3(4.0 * cis(2100.0 * t), 2830.0 * t), 1.0, 2.0).xy;

  //   dest += 0.2 * mix(0.5, 1.0, duck) * tanh(20.0 * env * wave);
  // }

  // { // shaker
  //   float t = mod(time.x, S2T);
  //   float st = mod(floor(time.y / S2T), 16.0);

  //   float vel = fract(st * 0.42 + 0.23);
  //   float env = smoothstep(0.0, 0.02, t) * exp(-exp2(5.0 - 2.0 * vel) * t);

  //   float phase = 280.0 * t;
  //   phase += phase + 0.1 * sin(TAU * phase);
  //   vec2 wave = shotgun(phase, 2.0, 0.4, exp2(mix(1.0, 3.0, vel)));

  //   dest += 0.1 * mix(0.3, 1.0, duck) * tanh(8.0 * env * wave);
  // }

  // { // open hihat
  //   float t = mod(time.x - 0.5 * B2T, B2T);
  //   float q = B2T - t;

  //   float env = exp2(-7.0 * t) * smoothstep(0.0, 0.01, q);

  //   vec2 sum = vec2(0.0);
  //   repeat(i, 16) {
  //     float odd = float(i % 2);
  //     float tt = (t + 0.3) * mix(1.0, 1.002, odd);
  //     vec3 dice = hash3f(vec3(i / 2));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * tt + dice2.xy);
  //     wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * tt + dice2.yz);
  //     wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * tt + dice2.zx);

  //     sum += wave * mix(1.0, 0.5, odd);
  //   }

  //   dest += 0.16 * env * duck * tanh(sum);
  // }

  // { // ride
  //   vec4 seq = seq16(time.y, 0x8888);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = exp2(-3.0 * t) * smoothstep(0.0, 0.01, q);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 2.9 * env * sin(wave + exp2(13.50 + 0.1 * dice.x) * t + dice2.xy);
  //     wave = 2.8 * env * sin(wave + exp2(12.97 + 0.2 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * env * sin(wave + exp2(14.09 + 0.1 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.06 * env * mix(0.3, 1.0, duck) * tanh(sum);
  // }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4100.0 * t, 1.9, 0.0, 1.0);
    dest += 0.6 * env * mix(0.3, 1.0, duck) * tanh(8.0 * wave);
  }

  { // snare roll
    float fade = smoothstep(32.0 * B2T, 64.0 * B2T, time.z);

    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    if (time.z > 60.0 * B2T) {
      float l = 0.125 * B2T;
      t = mod(time.x, l);
      q = l - t;
    }

    float env = smoothstep(0.0, 0.01, q);
    env *= mix(
      exp(-10.0 * max(t - 0.04, 0.0)),
      exp(-80.0 * t),
      0.3
    );

    float sinphase = 234.0 * t - 4.0 * exp2(-t * 200.0);
    float noisephase = 128.0 * t;
    vec2 wave = mix(
      mix(
        cis(TAU * (sinphase)),
        cis(TAU * (1.5 * sinphase)),
        0.3
      ),
      cheapnoise(noisephase) - cheapnoise(noisephase - 0.004),
      0.3
    );

    dest += 0.3 * p2 * fade * mix(0.5, 1.0, duck) * tanh(4.0 * env * wave);
  }

  if (time.z > 48.0 * B2T) { // sweep
    float t = tmod(time, 16.0 * B2T);

    float env = smoothstep(0.0, 16.0 * B2T, t);

    vec2 osc = vec2(0.0);

    { // noise
      osc += cheapnoise(128.0 * t);
      osc += cheapnoise(128.0 * (t + 0.002 * exp(-0.4 * t)));
      osc += cheapnoise(128.0 * (t + 0.004 * exp(-0.4 * t)));
    }

    { // saw
      float phase = 4.0 * exp(t / 4.0);
      phase += 0.002 * tri(t * 40.0);
      phase *= 600.0;
      osc += (2.0 * fract(phase) - 1.0);
    }

    dest += 0.1 * p1 * mix(0.4, 1.0, duck) * env * osc;
  }

  { // lead
    vec2 sum = vec2(0.0);
    repeat(i, 4) {
      vec4 tdelay = time - float(i) * B2T;
      float gs = t2sSwing(tmod(tdelay, timeLength.z));
      float tss = 0.0; // total start step
      float s = 0.0; // start step
      float sp = 0.0; // end step
      float pitch = 0.0;

      #define S(ds, sl, p) tss += float(ds); if (gs >= tss) { s = tss; sp = tss + float(sl); pitch = float(p); }
      S(0, 12, 7) S(12, 4, 0)
      S(4, 4, 12) S(4, 4, 11) S(4, 4, 9) S(4, 4, 11)
      S(4, 6, 9) S(6, 16, 7)
      S(18, 4, 4) S(4, 4, 5)

      S(4, 12, 7) S(12, 4, 0)
      S(4, 4, 5) S(4, 4, 4) S(4, 4, 2) S(4, 4, 4)
      S(4, 6, 2) S(6, 16, 0)
      S(18, 4, 4) S(4, 4, 5)

      S(4, 12, 7) S(12, 4, 0)
      S(4, 4, 12) S(4, 4, 11) S(4, 4, 9) S(4, 4, 11)
      S(4, 10, 7) S(12, 4, 0)
      S(4, 6, 11) S(6, 10, 12)

      S(66, 4, 4) S(4, 4, 5)
      #undef S

      float t = tmod(tdelay - s2tSwing(s), timeLength.z);
      float l = s2tSwing(sp) - s2tSwing(s);
      float q = l - t;

      t += 0.0004 * smoothstep(0.0, 0.5, t) * sin(TAU * 6.0 * t);

      vec3 dice = hash3f(vec3(s, 10, 30));

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

      pitch += 72.0 + TRANSPOSE;
      float freq = p2f(pitch);
      vec2 phase = freq * t + vec2(0.0, 0.3);

      vec2 osc = 2.0 * fract(phase) - 1.0;

      float delaydecay = exp2(-1.4 * float(i));
      sum += delaydecay * env * osc;
    }
    dest += p0 * 0.25 * mix(0.3, 1.0, duck) * sum;
  }

  // { // chord
  //   vec2 sum = vec2(0.0);
  //   repeat(iDelay, 4) {
  //     vec4 tdelay = mod(time - B2T * float(iDelay), timeLength);
  //     float delaydecay = exp2(-2.0 * float(iDelay));

  //     vec4 seq = seq16(tdelay.z, 0x9248);
  //     float t = seq.t;
  //     float q = seq.q;

  //     float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  //     env *= mix(
  //       smoothstep(0.0, 0.001, 2.0 * S2T - t),
  //       exp2(-t),
  //       0.3
  //     );

  //     repeat(iUnison, 40) {
  //       vec3 dice = hash3f(vec3(iUnison, seq.s, 7));
  //       vec2 dicen = boxMuller(dice.xy);

  //       float pitch = 60.0 + getChordNote(iUnison, tdelay.z);
  //       float freq = p2f(pitch) * exp2(0.02 * dicen.x);

  //       vec2 phase = t * freq * exp2(0.01 * (dice.xy - 0.5)) + dice.yx;

  //       float cutdecay = exp2(mix(4.0, -2.0, p4));
  //       float k = exp2(-cutdecay * t) * exp2(-0.1 * float(iDelay));
  //       vec2 wave = cheapfiltersaw(phase, k);
  //       wave += 0.3 * (2.0 * fract(2.0 * phase) - 1.0);

  //       sum += delaydecay * env * wave * rotate2D(2.4 * float(iUnison));
  //     }
  //   }

  //   dest += 0.04 * sum * mix(0.4, 1.0, duck);
  // }

  // { // pad, should I use this?
  //   vec2 sum = vec2(0.0);
  //   repeat(iUnison, 40) {
  //     vec3 dice = hash3f(vec3(iUnison, 7, 7));
  //     vec2 dicen = boxMuller(dice.xy);

  //     vec4 tdelay = mod(time - 0.2 * dice.y, timeLength);
  //     float l = 4.0 * B2T;
  //     float t = mod(tdelay.z, l);
  //     float st = round((tdelay.z - t) / S2T);

  //     float fade = smoothstep(0.0, S2T, t) * smoothstep(0.0, S2T, l - t);

  //     float pitch = 48.0 + getChordNote(iUnison, tdelay.z);
  //     float freq = p2f(pitch) * exp2(0.01 * dicen.x);

  //     float phase = t * freq + dice.x;

  //     vec3 p = vec3(4.0, 8.0, 5.0);
  //     p.x += 0.5 * fract(phase);
  //     p.y += 4.0 * fract(2.0 * phase);
  //     p.zx += 0.5 * cis(TAU * time.z / B2T / 32.0);

  //     vec2 wave = cyclic(p, 0.5, 2.0).xy;

  //     sum += fade * wave * rotate2D(2.4 * float(iUnison));
  //   }

  //   dest += 0.02 * mix(0.5, 1.0, duck) * sum;
  // }

  { // arp
    const int N_NOTES = 6;
    const int NOTES[] = int[](7, 12, 14, 19, 14, 12);

    const int N_NOTES_FILL = 8;
    const int NOTES_FILL[] = int[](7, 12, 14, 19, 23, 24, 26, 31);

    vec2 sum = vec2(0);
    repeat(i, 4) {
      vec4 tdelay = mod(time - float(i) * B2T, timeLength);
      float delaydecay = exp2(-1.0 * float(i));

      vec4 seq = seq16(tdelay.y, 0xffff);
      float st = seq.s + 16.0 * floor(tdelay.z / (16.0 * S2T));
      float t = seq.t;
      float q = seq.q;

      float pitch1 = float(NOTES[int(st) % 64 % N_NOTES]);
      float pitch0 = float(NOTES[(int(st) + N_NOTES - 1) % 64 % N_NOTES]);

      if (mod(tdelay.z / B2T, 32.0) > 31.0) {
        t = mod(tdelay.x, 0.5 * S2T);
        q = 0.5 * S2T - t;

        int iPitch = int(8.0 * (tdelay.x / B2T)) % N_NOTES_FILL;
        pitch1 = float(NOTES_FILL[iPitch]);
        pitch0 = iPitch == 0 ? pitch0 : float(NOTES_FILL[iPitch - 1]);
      }

      float basepitch = 60.0 + TRANSPOSE;
      pitch0 += basepitch;
      pitch1 += basepitch;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      env *= exp2(-6.0 * t);

      vec2 phase = vec2(glidephase(t, 0.01, pitch0, pitch1));
      phase += vec2(0.0, 0.1);

      vec2 osc = 2.0 * step(0.5 + 0.25 * tri(time.z / (S2T * 16.0)), fract(phase)) - 1.0;

      sum += delaydecay * env * osc;
    }

    dest += 0.08 * mix(0.4, 1.0, duck) * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  dest *= 0.8;
  return dest;
}
