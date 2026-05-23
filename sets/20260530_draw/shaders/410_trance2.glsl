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
#define TRANSPOSE 7.4

const float SWING = 0.5;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;

uniform vec4 param_knob2; // snare roll
uniform vec4 param_knob3; // kick cut
uniform vec4 param_knob4; // arp cutoff

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
  const int N_CHORD_NOTES = 8;
  const int CHORDS[] = int[](
    0, 7, 12, 12+2, 12+3, 12+5, 12+7, 12+10,
    -3, 7, 11, 12+2, 12+4, 12+7, 12+9, 12+11,
    -4, 7, 12, 12+3, 12+5, 12+7, 12+10, 24,
    1, 8, 10, 12+1, 12+5, 12+8, 12+10, 24
  );

  int iProg = 0;
  // { // enable progression
  //   iProg = imod(int(t / (8.0 * B2T)), 4);
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
      - 3.0 * exp2(-t * 10.0)
      - 8.0 * exp2(-t * 30.0)
      - 1.0 * exp2(-t * 80.0)
      - 4.0 * exp2(-t * 600.0)
      - 0.5 * exp2(-t * 80.0) * cheapnoise(2.0 * t + 5.0).x
    );

    vec2 wave = tanh(2.0 * sin(TAU * phase));

    dest += 0.7 * env * wave;
  }

  { // bass
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s;
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    float pitch = 24.0 + getChordNote(0, time.z);
    float freq = p2f(pitch);
    vec2 phase = vec2(freq * t);

    vec2 wave = cos(TAU * phase);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i, 20, 12));
      vec2 phaseu = phase * exp2(0.02 * (dice.x - 0.5)) + dice.yz;

      float fmamount = float[](0.0, 0.5, 1.0, 0.8)[int(seq.s) % 4];
      float fmenv = exp2(-20.0 * t);
      phaseu += 0.1 * fmamount * fmenv * sin(3.0 * TAU * phase) * vec2(1, -1);

      float k = exp2(-3.0 * t);
      wave += 0.5 * cheapfiltersaw(2.0 * phaseu, k);
    }

    dest += 0.4 * mix(0.0, 1.0, duck) * env * wave;
  }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-exp2(7.0 - 1.0 * fract(seq.s * 0.61 + 0.6)) * t);

  //   vec2 wave = shotgun(3200.0 * t, 2.4, 0.0, 1.0);
  //   wave = tanh(6.0 * wave);

  //   dest += 0.4 * mix(0.1, 1.0, duck) * env * wave;
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

  //   dest += 0.2 * env * duck * tanh(sum);
  // }

  // { // snare roll
  //   vec4 seq = seq16(time.y, 0x0808);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp(-20.0 * max(t - 0.04, 0.0));

  //   float sinphase = 224.0 * t - 4.0 * exp2(-t * 200.0);
  //   float noisephase = 128.0 * t;
  //   vec2 wave = mix(
  //     mix(
  //       cis(TAU * (sinphase)),
  //       cis(TAU * (1.5 * sinphase)),
  //       0.4
  //     ),
  //     cheapnoise(noisephase) - cheapnoise(noisephase - 0.004),
  //     0.3
  //   );

  //   dest += 0.3 * mix(0.5, 1.0, duck) * tanh(8.0 * env * wave);
  // }

  // { // shaker
  //   float t = mod(time.x, S2T);
  //   float st = mod(floor(time.y / S2T), 8.0);

  //   float vel = fract(st * 0.39 + 0.25);
  //   float env = smoothstep(0.0, 0.02, t) * exp(-exp2(5.0 - 2.0 * vel) * t);

  //   float phase = 160.0 * t;
  //   phase += phase + 0.1 * sin(TAU * phase);
  //   vec2 wave = shotgun(phase, 3.0, 0.3, exp2(mix(0.0, 3.0, vel)));

  //   dest += 0.2 * mix(0.3, 1.0, duck) * tanh(8.0 * env * wave);
  // }

  // { // ride
  //   vec4 seq = seq16(time.y, 0xaaaa);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = exp2(-4.0 * t) * smoothstep(0.0, 0.01, q);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 2.9 * env * sin(wave + exp2(12.50 + 0.5 * dice.x) * t + dice2.xy);
  //     wave = 2.8 * env * sin(wave + exp2(12.97 + 0.5 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * env * sin(wave + exp2(14.39 + 0.5 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.12 * env * mix(0.3, 1.0, duck) * tanh(sum);
  // }

  // { // crash
  //   float t = mod(time.z, 64.0 * B2T);

  //   float env = mix(exp(-t), exp(-10.0 * t), 0.7);
  //   vec2 wave = shotgun(4100.0 * t, 1.9, 0.0, 1.0);
  //   dest += 0.5 * env * mix(0.2, 1.0, duck) * tanh(8.0 * wave);
  // }

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

    dest += 0.3 * p2 * fade * mix(0.3, 1.0, duck) * tanh(4.0 * env * wave);
  }

  { // arp
    vec2 sum = vec2(0);
    repeat(i, 64) {
      float fi = float(i);
      float iDelay = mod(fi, 4.0);

      vec3 dice = hash3f(vec3(fi, 48, 48));

      vec4 tdelay = mod(time - 3.0 * S2T * iDelay, timeLength);

      vec4 seq = seq16(tdelay.y, 0xffff);
      float st = seq.s + 16.0 * floor(tdelay.z / S2T / 16.0);
      float t = seq.t;
      float q = seq.q;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      env *= exp2(-5.0 * t);

      int iNote = int(16.0 * fract(0.424 * st + 0.6));
      float pitch = 36.0 + getChordNote(iNote, tdelay.z);
      pitch += float(iNote / 8 * 12);
      float freq = p2f(pitch);
      vec2 phase = t * freq + vec2(0.5, 0.0);
      phase *= exp2(0.1 * (dice.xy - 0.5)); // detune
      phase += dice.yz;

      float cutdecay = exp2(mix(4.0, -2.0, p4));
      float k = exp2(-cutdecay * t) * exp2(-0.1 * iDelay);
      float delaydecay = exp2(-1.0 * iDelay);
      sum += env * delaydecay * (
        cheapfiltersaw(phase, k)
      );
    }

    dest += 0.08 * mix(0.3, 1.0, duck) * sum;
  }

  // { // chord
  //   vec4 seqg = seq16(time.y, 0xffff);

  //   vec2 sum = vec2(0.0);
  //   repeat(iUnison, 64) {
  //     vec3 dice = hash3f(vec3(iUnison, 7, 7));
  //     vec2 dicen = boxMuller(dice.xy);

  //     vec4 tdelay = mod(time - 0.2 * dice.y, timeLength);
  //     float l = 16.0 * B2T;
  //     float t = mod(tdelay.z, l);
  //     float st = round((tdelay.z - t) / S2T);

  //     float fade = smoothstep(0.0, S2T, t) * smoothstep(0.0, S2T, l - t);

  //     float pitch = 48.0 + getChordNote(iUnison, tdelay.z);
  //     float freq = p2f(pitch) * exp2(0.015 * dicen.x);
  //     vec2 phase = t * freq + dice.xy;

  //     vec2 wave = cheapfiltersaw(phase, 0.98);

  //     sum += fade * wave * rotate2D(2.4 * float(iUnison));
  //   }

  //   dest += 0.02 * sum * mix(0.4, 1.0, duck);
  // }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  dest *= 0.8;
  return dest;
}
