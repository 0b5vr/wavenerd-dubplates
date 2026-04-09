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
#define TRANSPOSE 6.0

const float SWING = 0.5;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

uniform vec4 param_knob0;

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

float dotNoise(vec3 p) {
  mat3 R = orthBas(vec3(3, 4, -1));
  return dot(cos(R * p), sin(1.618 * p * R)) / 3.0;
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

vec3 corruptor(float t, float corrupt) {
  const float PERS = 0.5;
  const float LACU = 1.4;

  float n = cyclic(vec3(t, corrupt, 0.0), PERS, LACU).x;
  n = floor(n + corrupt);
  return hash3f(vec3(t, 0.0, n));
}

float cheapfiltersaw(float phase, float k) {
  float wave = fract(phase);
  float c = smoothstep(1.0, 0.0, wave / (1.0 - k));
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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float duck = 1.0;

  { // kick
    int ptn = int(time.z / timeLength.y) % 2;
    vec4 seq = seq16(time.y, int[](0x8020, 0x8124)[ptn]);
    float t = seq.t;
    float q = seq.q;

    {
      float env = smoothstep(0.0, 0.001, q);
      env *= smoothstep(0.3, 0.1, t);

      // { // highpass-like
      //   env *= exp(-50.0 * t);
      // }

      float phase = (
        42.0 * t
        - 5.0 * exp2(-t * 20.0)
        - 5.0 * exp2(-t * 100.0)
        - 9.0 * exp2(-t * 200.0)
      );

      float wave = sin(TAU * phase);

      dest += 0.5 * env * wave;
    }

    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );
  }

  { // snare
    vec4 seq = seq16(time.y, 0x0808);
    float t = seq.t;
    float q = seq.q;

    {
      float zc = smoothstep(0.0, 0.01, q);

      vec2 noise = cheapnoise(256.0 * t) - cheapnoise(256.0 * t - 0.012);
      noise *= mix(
        exp(-100.0 * t),
        exp(-40.0 * max(t - 0.1, 0.0)),
        0.7
      );

      float bodyphase = 220.0 * t - 10.0 * exp(-400.0 * t);
      vec2 body = vec2(sin(TAU * bodyphase));
      body *= exp(-80.0 * t);

      float fxphase = 880.0 * t;
      fxphase += tri(40.0 * t);
      float fx = step(0.8, fract(fxphase));
      fx *= exp(-20.0 * t);

      vec2 wave = mix(body, noise, 0.4);
      wave += 0.4 * fx;
      wave = tanh(8.0 * wave);

      float penv = mix(exp(-30.0 * t), 1.0, 0.5);
      dest += 0.32 * mix(0.5, 1.0, duck) * zc * penv * wave;
    }

    duck = min(
      duck,
      smoothstep(0.0, 0.2, t) * smoothstep(0.0, 0.001, q)
    );
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s;
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);

    float vel = fract(0.429 * st + 0.32);
    env *= exp(-exp2(8.0 - 3.0 * vel) * t);

    vec2 wave = shotgun(4700.0 * t, 1.8, 0.0, 0.0);
    wave = tanh(1.5 * wave);

    dest += 0.4 * mix(0.1, 1.0, duck) * env * wave;
  }

  { // beep
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s + lofi(time.z / S2T, 16.0);
    float t = seq.t;
    float q = seq.q;

    float st8th = lofi(st, 2.0);
    vec3 dice8th = hash3f(vec3(st8th, 20, 11));
    bool retrigger = dice8th.x < 0.2;
    if (retrigger) {
      st = st8th;

      float l = exp2(-floor(3.0 * dice8th.y)) * S2T;
      t = mod(time.x, l);
      q = l - t;
    }

    vec3 dice = hash3f(vec3(st, 20, 70));
    vec3 dice2 = hash3f(vec3(st, 21, 40));

    float env = smoothstep(0.0, 0.01, q);

    float freq = exp2(mix(6.0, 11.0, dice.x));
    float fall = exp2(mix(-8.0, 0.0, pow(dice.y, 6.0)));
    float phase = freq * exp(-t / fall) * fall;

    vec3 p = vec3(
      exp2(-1.0 + 4.0 * dice2.x) * cis(TAU * phase),
      20.0 * dice2.y + phase * exp2(mix(-8.0, 0.0, pow(dice2.z, 3.0)))
    );
    vec2 wave = cyclic(p, 0.5, 2.0).xy;

    dest += 0.1 * mix(0.1, 1.0, duck) * env * wave;
  }

  { // ride
    vec4 seq = seq16(time.y, 0x8692);
    float st = seq.s;
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp(-10.0 * t);

    vec2 wave = shotgun(4300.0 * t, 2.8, 0.4, 0.6);
    wave = tanh(1.5 * wave);

    dest += 0.4 * mix(0.1, 1.0, duck) * env * wave;
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4100.0 * t, 1.9, 0.0, 1.0);
    dest += 0.2 * env * mix(0.1, 1.0, duck) * tanh(8.0 * wave);
  }

  { // chord stuff
    const int CHORD_NOTES_N = 8;
    const int CHORD_PROG_N = 4;
    const int CHORDS[] = int[](
      -4, 3, 7, 12, 14, 15, 19, 22,
      -3, 4, 9, 11, 12, 14, 16, 19,
      -4, 3, 7, 10, 12, 14, 15, 19,
      0, 7, 10, 12, 14, 15, 19, 22
    );
    const int CHORD_PROGTRANS[] = int[](0, 0, 0, 0);
    // const int CHORD_PROGTRANS[] = int[](2, 0, 0, -1);

    #define CHORD_NOTE(i, st) (TRANSPOSE + float(CHORDS[i % CHORD_NOTES_N + int(st) / 32 % CHORD_PROG_N * CHORD_NOTES_N] + CHORD_PROGTRANS[int(st) / 32 % CHORD_PROG_N]))

    { // bass
      float st1 = t2sSwing(time.y) + lofi(time.z / S2T, 16.0);
      float st0 = mod(st1 - 1.0, 256.0);
      float t = time.y;
      float tst = t - s2tSwing(mod(st1, 16.0));

      vec3 dice0 = hash3f(vec3(st0, 19, 31));
      vec3 dice1 = hash3f(vec3(st1, 19, 31));

      float gate0 = step(dice0.x, float[](0.04, 0.8, 0.6, 0.8)[int(st0) % 4]);
      float gate1 = step(dice1.x, float[](0.04, 0.8, 0.6, 0.8)[int(st1) % 4]);
      float gate = mix(gate0, gate1, smoothstep(0.0, 0.004, tst));

      float pitch = 24.0 + CHORD_NOTE(0, st1);
      float freq = p2f(pitch);
      float phase = freq * t;
      phase -= 0.01 * fract(120.0 * phase);

      vec2 wave = vec2(sin(TAU * phase));
      dest += 0.7 * mix(0.0, 1.0, duck) * gate * wave;
    }

    { // chord
      vec2 sum = vec2(0.0);

      vec4 seq = seq16(time.y, 0x9248);
      float st = seq.s + lofi(time.z / S2T, 16.0);
      float t = seq.t;
      float q = seq.q;

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      env *= mix(
        smoothstep(3.0 * S2T, 2.5 * S2T, t),
        exp2(-5.0 * t),
        0.3
      );

      repeat(i, 8) {
        float fi = float(i);
        vec3 dice = hash3f(vec3(i, 64, 11));

        float note = 48.0 + CHORD_NOTE(i, st);
        float freq = p2f(note);
        float phase = t * freq + dice.z;
        phase += 0.004 * freq * sin(time.z + TAU * dice.y);

        phase += 0.2 * sin(TAU * phase);

        vec2 wave = vec2(cheapfiltersaw(phase, 0.99));

        sum += vec2(wave) * rotate2D(fi) / 32.0;
      }

      dest += 0.8 * mix(0.1, 1.0, duck) * env * sum;
    }

    { // arp
      vec2 sum = vec2(0.0);

      repeat(i, 32) {
        float fi = float(i);

        float toff = s2tSwing(fi);
        float st1 = lofi((time.z - toff) / S2T, 32.0) + fi;
        float st0 = mod(st1 - 1.0, 256.0);
        float t = tmod(time - toff, 32.0 * S2T);
        float q = timeLength.y - t;
        vec3 dice = hash3f(vec3(i, 81, 7));

        float l2 = 3.0 * S2T;
        float t2 = mod(t, l2);
        float q2 = min(q, l2 - t2);
        float delaycount = floor(t / l2);

        float env = smoothstep(0.0, 0.001, t2) * smoothstep(0.0, 0.001, q2);
        env *= smoothstep(1.2 * S2T, 1.0 * S2T, t2);

        int arpSrc0 = int(fract(0.131 * st0) * float(3 * CHORD_NOTES_N));
        int arpSrc1 = int(fract(0.131 * st1) * float(3 * CHORD_NOTES_N));
        int proghead0 = int(st0) / 32 % CHORD_PROG_N * CHORD_NOTES_N;
        int proghead1 = int(st1) / 32 % CHORD_PROG_N * CHORD_NOTES_N;
        float pitch0 = 36.0 + 12.0 * float(arpSrc0 / CHORD_NOTES_N) + CHORD_NOTE(arpSrc0, st0);
        float pitch1 = 36.0 + 12.0 * float(arpSrc1 / CHORD_NOTES_N) + CHORD_NOTE(arpSrc1, st1);
        float phase = glidephase(t2, 0.01, pitch0, pitch1) * exp2(0.02 * (dice.x - 0.5));

        float k = 0.99 * env * exp2(-12.0 * t2);
        vec2 wave = vec2(
          cheapfiltersaw(phase * exp2(-0.004), k),
          cheapfiltersaw(phase * exp2(0.004), k)
        );

        sum += exp2(-0.7 * delaycount) * env * wave * rotate2D(delaycount);
      }

      dest += 0.1 * mix(0.1, 1.0, duck) * sum;
    }
  }

  return clip(1.3 * tanh(dest));
}
