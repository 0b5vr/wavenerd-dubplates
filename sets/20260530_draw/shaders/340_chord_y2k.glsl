// - enable sweep
// - early return the chord riff to solo
// - substitute the chord riff to play chord + arp + fx
// - DON'T FORGET TO ENABLE THE BASS CHORD PROGRESSION
// - drop the kick and bass
// - enable fill in
// - unmute everything
// - 😀👍

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
#define TRANSPOSE 3.0

const float SWING = 0.57;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

uniform vec4 param_knob1; // sweep
uniform vec4 param_knob2; // snare roll
uniform vec4 param_knob3; // kick cut

#define p1 paramFetch(param_knob1)
#define p2 paramFetch(param_knob2)
#define p3 paramFetch(param_knob3)

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

float s2tSwing(int st) {
  return s2tSwing(float(st));
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

vec2 mainAudioDry(vec4 time) {
  vec2 dest = vec2(0.0);

  float duck = smoothstep(0.0, 0.4, time.x) * smoothstep(0.0, 0.001, B2T - time.x);
  bool isFillIn = false;
  // { // enable fill-in
  //   isFillIn = time.z > 60.0 * B2T;
  // }

  const int N_CHORD_NOTES = 8;
  const int N_CHORD_PROGS = 4;
  const int CHORDS[] = int[](
    0, 7, 10, 12, 14, 17, 19, 22,
    -3, 4, 7, 12, 14, 16, 19, 23,
    -4, 3, 7, 10, 12, 15, 19, 24,
    1, 8, 10, 12, 15, 17, 19, 24
  );

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    if (isFillIn) {
      seq = seq16(time.y, 0x8972);
    }

    float t = seq.t;
    float q = seq.q;
    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    float env = smoothstep(0.0, 0.001, q);
    env *= smoothstep(0.25, 0.1, t);
    env *= mix(1.0, exp2(-80.0 * t), p3); // hpf-like

    float phase = (
      52.0 * t
      - 4.0 * exp2(-t * 20.0)
      - 4.0 * exp2(-t * 80.0)
      - 3.0 * exp2(-t * 400.0)
    );

    float wave = sin(1.7 * sin(TAU * phase));

    dest += 0.6 * env * wave;
  }

  { // rim
    vec4 seq = seq16(time.y, 0x5657);
    float t = seq.y;

    float env = exp2(-400.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * mix(0.3, 1.0, duck) * vec2(wave) * rotate2D(seq.x);
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp2(-exp2(7.0 - 3.0 * fract(seq.s * 0.41 + 0.6)) * t);

    vec2 wave = shotgun(4700.0 * t, 1.8, 0.0, 0.0);
    wave = tanh(1.5 * wave);

    dest += 0.25 * mix(0.1, 1.0, duck) * env * wave;
  }

  // { // open hihat
  //   vec4 seq = seq16(time.y, 0x2222);
  //   float t = seq.y;

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.30 + 0.1 * dice.x) * t + dice2.xy);
  //     wave = 3.2 * exp2(-2.0 * t) * sin(wave + exp2(11.78 + 0.3 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * exp2(-35.0 * t) * sin(wave + exp2(14.92 + 0.2 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.13 * mix(0.0, 1.0, duck) * tanh(2.0 * sum);
  // }

  // { // clap
  //   vec4 seq = seq16(time.y, 0x0808);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = mix(
  //     exp2(-80.0 * t),
  //     exp2(-500.0 * mod(t, 0.012)),
  //     exp2(-100.0 * max(0.0, t - 0.02))
  //   );

  //   vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0).xy;

  //   dest += 0.2 * tanh(20.0 * env * wave);
  // }

  // { // snare
  //   vec4 seq = seq16(time.y, 0x2162);
  //   if (isFillIn) {
  //     seq = seq16(time.y, 0x257f);
  //   }

  //   float t = seq.t;
  //   float q = seq.q;
  //   if (isFillIn && time.y > 3.5 * B2T) {
  //     t = mod(time.x, 0.125 * B2T);
  //     q = t + 0.125 * B2T;
  //   }

  //   float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
  //   env *= exp(-14.0 * max(t - 0.05, 0.0));

  //   float phase = 190.0 * t;
  //   phase += 4.0 * (1.0 - exp2(-t * 200.0));

  //   vec2 wave = mix(
  //     mix(
  //       cis(TAU * phase),
  //       cis(1.5 * TAU * phase),
  //       0.4
  //     ),
  //     cheapnoise(128.0 * t) - cheapnoise(128.0 * t - 0.008),
  //     0.3
  //   );

  //   float amp = 0.2 * mix(0.04, 1.0, duck);
  //   if (isFillIn) { amp = 0.25; }
  //   dest += amp * env * wave;
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
  //     wave = 2.9 * env * sin(wave + exp2(13.10 + 0.4 * dice.x) * t + dice2.xy);
  //     wave = 2.8 * env * sin(wave + exp2(14.97 + 0.4 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * env * sin(wave + exp2(14.09 + 1.0 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.03 * env * mix(0.3, 1.0, duck) * tanh(sum);
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

    float sinphase = 220.0 * t - 3.0 * exp2(-t * 200.0);
    float noisephase = 100.0 * t;
    vec2 wave = mix(
      mix(
        cis(TAU * (sinphase)),
        cis(TAU * (1.5 * sinphase)),
        0.3
      ),
      cheapnoise(noisephase) - cheapnoise(noisephase - 0.004),
      0.3
    );

    dest += 0.2 * p2 * fade * mix(0.5, 1.0, duck) * tanh(8.0 * env * wave);
  }

  if (time.z > 48.0 * B2T) { // sweep
    float l = 16.0 * B2T;
    float t = tmod(time, l);
    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, l - t);
    env *= pow(t / l, 2.0);

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      vec3 dice = hash3f(vec3(i / 4, 20, 228));

      float tt = t * (1.0 + 0.0006 * float(i % 4));

      float phase = 200.0 * (exp(tt / 4.0) * 4.0) * exp2(0.1 * dice.x);
      phase += sin(TAU * phase * exp2(0.5 * dice.y));
      phase += 0.2 * sin(6.0 * TAU * phase * exp2(dice.z));
      sum += cis(TAU * phase) / 16.0;
    }

    dest += 0.32 * p1 * mix(0.4, 1.0, duck) * env * sum;
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);
    if (isFillIn) {
      t = time.y;
    }

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4500.0 * t, 1.4, 0.0, 1.0);
    dest += 0.4 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // bass
    float l = 4.0 * B2T;
    float t = time.y;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, l - t);

    int iProg = 0;
    // { // progression
    //   iProg = imod(int(time.z / (4.0 * B2T)), N_CHORD_PROGS);
    // }

    int iChord = iProg * N_CHORD_NOTES;
    float pitch = 24.0 + TRANSPOSE + float(CHORDS[iChord]);
    float freq = p2f(pitch);
    float phase = freq * t;

    vec2 wave = vec2(sin(2.0 * sin(TAU * phase)));
    dest += 0.5 * mix(0.0, 1.0, duck) * wave;
  }

  // { // chord riff
  //   vec2 sum = vec2(0.0);
  //   repeat(iUnison, 64) {
  //     repeat(iDelay, 3) {
  //       vec3 dice = hash3f(vec3(iUnison, iDelay, 50));
  //       float tOsc = tmod(time - 4.0 * float(iDelay) * S2T, timeLength.z);
  //       int iProg = imod(int(floor((tOsc - S2T) / (4.0 * B2T))), + N_CHORD_PROGS);
  //       tOsc = tmod(time - 4.0 * float(iDelay) * S2T, timeLength.y);

  //       float tGateStart = 0.0;
  //       float tGateEnd = 0.0;
  //       #define S(x) s2tSwing(x)
  //       #define SEQ(a,b) if(S(a)<tOsc){tGateStart=S(a);tGateEnd=S(b);}
  //       SEQ(11-16,12-16)
  //       SEQ(1,3.2)
  //       SEQ(4,5)
  //       SEQ(6,7)
  //       SEQ(8,10.2)
  //       SEQ(11,12)
  //       float tGate = tOsc - tGateStart;
  //       float lGate = tGateEnd - tGateStart;
  //       float envGate = smoothstep(0.0, 0.001, tGate) * mix(
  //         smoothstep(0.0, 0.01, lGate - tGate),
  //         exp2(6.0 * min(lGate - tGate, 0.0)),
  //         0.1
  //       );

  //       int iChord = iProg * N_CHORD_NOTES + iUnison % 8;
  //       vec2 pitch = 48.0 + TRANSPOSE + float(CHORDS[iChord]) + 0.05 * boxMuller(dice.xy);
  //       vec2 freq = p2f(pitch);
  //       vec2 phase = freq * mod(tGate, timeLength.y) + hash3f(dice).xy;

  //       vec2 wave = vec2(0.0);

  //       float k = 0.99 * exp2(-0.01 * tGate) * exp2(-0.01 * float(iDelay));
  //       wave += cheapfiltersaw(phase, k);

  //       sum += envGate * vec2(wave) * exp2(-2.0 * float(iDelay));
  //     }
  //   }
  //   dest += 1.3 * mix(0.4, 1.0, duck) * sum / 32.0;
  // }

  { // arp
    int N_PITCH = 4;
    int PITCHES[]=int[](0, 7, 10, 5);

    vec2 sum = vec2(0);
    repeat(iDelay, 3) {
      vec4 timeDelayed = mod(time - float(iDelay) * B2T, timeLength);
      vec4 seq = seq16(timeDelayed.y, 0xffff);
      float t = seq.t;
      float q = seq.q;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
      env *= exp2(-40.0 * t);

      float i1 = 16.0 * floor(time.z / 4.0 / B2T) + seq.s - 4.0 * float(iDelay);
      int iNote1 = int(fract(i1 * 0.64) * float(N_PITCH));
      float pitch1 = 48.0 + TRANSPOSE + float(PITCHES[iNote1]) + 12.0 * floor(3.0 * fract(i1 * 0.43));

      float i0 = i1 - 1.0;
      int iNote0 = int(fract(i0 * 0.64) * float(N_PITCH));
      float pitch0 = 48.0 + TRANSPOSE + float(PITCHES[iNote0]) + 12.0 * floor(3.0 * fract(i0 * 0.43));

      float phase = glidephase(t, 0.03, pitch0, pitch1);
      vec2 wave = sin(TAU * phase + vec2(0.0, 0.8));
      sum += exp(-1.5 * float(iDelay)) * tanh(6.0 * sin(exp2(2.0 + sin(3.0 * timeDelayed.w)) * env * wave));
    }

    dest += 0.05 * mix(0.4, 1.0, duck) * sum;
  }

  { // fx
    vec2 sum = vec2(0.0);
    repeat(iDelay, 4) {
      float t = tmod(time - 2.0 * B2T * float(iDelay), 32.0 * B2T);

      float env = smoothstep(0.0, 0.001, t) * smoothstep(4.0, 1.0, t);

      float lfo = sin(TAU * 8.0 * t);

      float pitch0 = 67.0 + TRANSPOSE;
      float pitch1 = pitch0 + 12.0;
      float phase = glidephase(t, 0.5, pitch0, pitch1);

      vec2 wave = vec2(sin(3.0 * sin(TAU * phase))) * rotate2D(2.0 * TAU * t);
      wave *= lfo;
      sum += exp2(-float(iDelay)) * env * wave;
    }

    dest += 0.05 * mix(0.2, 1.0, duck) * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  return dest;
}
