#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float SWING = 0.64;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;

uniform vec4 param_knob1; // sweep
uniform vec4 param_knob2; // snare roll
uniform vec4 param_knob3; // kick cut

#define p1 paramFetch(param_knob1)
#define p2 paramFetch(param_knob2)
#define p3 paramFetch(param_knob3)

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

float p2f(float p) {
  return exp2((p - 69.0) / 12.0) * 440.0;
}

float f2p(float f) {
  return 12.0 * log2(max(f, 1.0) / 440.0) + 69.0;
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

vec2 mainAudioDry(vec4 time) {
  vec2 dest = vec2(0);
  float duck = smoothstep(0.0, 0.4, time.x) * smoothstep(0.0, 0.001, B2T - time.x);

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    float env = smoothstep(0.0, 0.001, q) * exp(-20.0 * max(t - 0.1, 0.0));
    env *= mix(1.0, exp2(-80.0 * t), p3); // hpf-like

    float tt = t;
    float wave = 0.0;

    wave += tanh(1.5 * sin(TAU * (
      40.0 * tt
      - 9.0 * exp2(-tt * 33.0)
      - 2.0 * exp2(-tt * 77.0)
    )));

    float tick = env * exp2(-500.0 * t);
    wave += env * tanh(1.5 * sin(TAU * 3.0 * tick));

    dest += 0.6 * env * wave;
  }

  { // hihat
    float st;
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = exp(-exp2(5.0) * t) * smoothstep(0.0, 0.01, q);
    vec2 wave = shotgun(3700.0 * t, 2.4, 0.0, 1.0);
    dest += 0.13 * env * duck * tanh(8.0 * wave);
  }

  { // clap
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

  { // shaker
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.y;
    float q = seq.w;
    float st = mod(seq.s, 8.0);

    float vel = fract(st * 0.42 + 0.2);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 1.4 * vel) * t);

    float phase = 100.0 * t;
    phase += phase + 0.01 * sin(TAU * phase); // fm
    vec2 wave = shotgun(phase, 3.0, 0.1, exp2(mix(1.0, 3.0, vel)));

    dest += 0.16 * env * duck * tanh(8.0 * wave);
  }

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
      vec3 dice = hash3f(vec3(i, 20, 228));

      float phase = 200.0 * (exp(t / 4.0) * 4.0) * exp2(0.5 * dice.x);
      phase += sin(TAU * phase * exp2(0.2 * dice.y));
      phase += 0.1 * sin(6.0 * TAU * phase * exp2(0.2 * dice.z));
      sum += cis(TAU * phase) / 16.0;
    }

    dest += 0.32 * p1 * mix(0.4, 1.0, duck) * env * sum;
  }

  { // crash
    float t = mod(time.z, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(4200.0 * t, 2.0, 0.0, 0.0);
    dest += 0.3 * env * mix(0.2, 1.0, duck) * tanh(8.0 * wave);
  }

  // { // 42 melody
  //   const int N_CHORD = 8;
  //   const int N_PROG = 2;
  //   const int CHORDS[N_CHORD * N_PROG] = int[](
  //     -12, 0, 7, 12, 14, 16, 19, 24,
  //     -12, 3, 7, 10, 14, 15, 19, 24
  //   );

  //   float st8 = floor((time.z + 0.5 * B2T) / (0.25 * B2T));
  //   float ot = float(int(st8) & 42);
  //   float prog = mod(floor(st8 / 32.0), 2.0);
  //   int chordOff = int(prog) * N_CHORD;

  //   float l = 0.5 * B2T;
  //   float t = mod(time.x, l);
  //   float q = l - t;

  //   float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);
  //   env *= step(0.5, ot);

  //   float bfreq = 8000.0 / 256.0;
  //   bfreq *= ot;
  //   float bpitch = f2p(bfreq);

  //   float mpitch = mod(bpitch + 6.0, 12.0) - 6.0;

  //   { // original
  //     float freq = p2f(bpitch + TRANSPOSE);
  //     float phase = freq * lofi(t, 1.0 / 8000.0);

  //     vec2 wave = 2.0 * fract(phase + vec2(0.0, 0.0)) - 1.0;
  //     dest += 0.12 * mix(1.0, 1.0, duck) * env * wave;
  //   }

  //   { // bass
  //     float pitch = 36.0 + TRANSPOSE + mpitch;
  //     float freq = p2f(pitch);
  //     float phase = freq * t;

  //     { // sub bass
  //       float wave = tanh(2.0 * sin(TAU * phase));
  //       dest += 0.4 * duck * env * wave;
  //     }

  //     { // fm bass
  //       float fmenv = smoothstep(0.0, 0.04, t) * exp(-10.0 * t);
  //       float fmot = mix(4.0, 8.0, prog);
  //       float fm = 8.0 * fmenv * sin(fmot * TAU * phase);
  //       vec2 wave = cis(TAU * phase + fm + time.z);
  //       dest += 0.12 * duck * env * wave;
  //     }
  //   }

  //   { // choir
  //     vec2 sum = vec2(0.0);

  //     repeat(i, 64) {
  //       float fi = float(i);
  //       vec3 dice = hash3f(float(i) + vec3(8, 4, 5));

  //       float pitch = 48.0 + TRANSPOSE + mpitch + float(CHORDS[i % N_CHORD + chordOff]);
  //       float freq = p2f(pitch) * exp2(0.016 * tan(2.0 * dice.y - 1.0));
  //       float phase = t * freq;

  //       vec3 c = vec3(0.0);
  //       vec3 d = mix(
  //         vec3(1.0, 1.0, -6.0),
  //         vec3(2.0, -3.0, -8.0),
  //         prog
  //       );
  //       float k = 0.5;
  //       vec2 wave = cyclic(fract(phase) * d, k, 2.0).xy;

  //       sum += vec2(wave) * rotate2D(2.5 * fi);
  //     }

  //     dest += 0.4 * mix(0.2, 1.0, duck) * env * tanh(0.12 * sum);
  //   }

  //   { // arp
  //     int iarp = int(16.0 * t / B2T);
  //     float pitch = 48.0 + TRANSPOSE + mpitch + float(CHORDS[iarp % N_CHORD + chordOff]) + 12.0 * float((iarp % 3) / 2);
  //     float freq = p2f(pitch);
  //     float phase = TAU * lofi(t * freq, 1.0 / 16.0);

  //     float z = mix(14.0, 3.0, prog);
  //     vec2 wave = cyclic(vec3(cis(phase), z), 0.5, 2.0).xy * rotate2D(TAU * 2.0 * time.w);

  //     dest += 0.2 * duck * env * wave;
  //   }
  // }

  // dest = mix(vec2(dot(dest, vec2(0.5))), dest, 0.0);
  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  dest *= 1.2;
  return dest;
}
