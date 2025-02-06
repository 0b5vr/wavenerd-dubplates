#define TRANSPOSE -5.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract((p)+0.25)-0.5))
#define u2b(u) ((u) * 2.0 - 1.0)
#define b2u(b) ((b) * 0.5 + 0.5)
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float SWING = 0.52;

const float LN2 = log(2.0);
const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float P5 = pow(2.0, 7.0 / 12.0);

uniform vec4 param_knob4; // pluck fm modulator

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

vec4 quant(float t, float interval, out float i) {
  interval = max(interval, 1.0);
  float st = t2sSwing(t);

  i = floor(floor(st) / interval);

  float prevStep = ceil(i * interval);
  float prevTime = s2tSwing(prevStep);
  float nextStep = ceil((i + 1.0) * interval);
  float nextTime = s2tSwing(nextStep);

  return vec4(
    prevStep,
    t - prevTime,
    nextStep,
    nextTime - t
  );
}

vec4 quant(float t, float interval) {
  float _;
  return quant(t, interval, _);
}

float p2f(float p) {
  return exp2((p - 69.0) / 12.0) * 440.0;
}

float cheapFilterSaw(float phase, float k) {
  float wave = fract(phase);
  float c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
}

vec2 cheapFilterSaw(vec2 phase, float k) {
  vec2 wave = fract(phase);
  vec2 c = smoothstep(1.0, 0.0, wave / (1.0 - k));
  return (wave + c - 1.0) * 2.0 + k;
}

vec2 ladderLPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 4.0 * omega * (omegaSq - 1.0);
  float b = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;

  return vec2(
    1.0 / sqrt(a * a + b * b),
    atan(a, b)
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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float duck = smoothstep(0.0, 0.4, time.x) * smoothstep(0.0, 0.001, B2T - time.x);

  const int N_CHORD = 8;
  int CHORDS[N_CHORD * 2] = int[](
    -11, 1, 8, 10, 15, 17, 24, 31,
    -17, -5, 2, 5, 9, 12, 14, 24
  );

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    float env = smoothstep(0.0, 0.001, q) * exp(-20.0 * max(t - 0.1, 0.0));

    // {
    //   env *= exp(-50.0 * t);
    // }

    float phase = TAU * (
      48.0 * t
      - 2.0 * exp2(-t * 40.0)
      - 3.0 * exp2(-t * 90.0)
    );

    float wave = tanh(1.5 * sin(phase));

    float tick = env * exp2(-500.0 * t);
    wave += env * tanh(1.5 * sin(TAU * 3.0 * tick));

    dest += 0.6 * env * wave;
  }

  { // rumble
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.04, q);

    vec2 phase = TAU * vec2(
      48.0 * t
      + 0.2 * cheapnoise(1.0 * t)
      + 0.06 * cheapnoise(2.0 * t)
    );
    vec2 wave = tanh(1.5 * sin(phase));

    dest += 0.4 * mix(0.0, 1.0, duck) * env * wave;
  }

  { // bass
    vec4 seq = seq16(time.y, 0xffff);
    float s = mod(seq.s, 8.0);
    float t = seq.t;
    float q = seq.q;

    vec3 dice = hash3f(vec3(s, 721, 124));

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

    float freq = exp2(4.8 + 2.0 * dice.x);
    float phase = freq * t;
    vec2 wave = vec2(tanh(3.0 * cheapFilterSaw(phase, 0.5)));

    dest += 0.4 * mix(0.0, 1.0, duck) * env * wave;
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = exp2(-80.0 * t) * smoothstep(0.0, 0.01, q);
    vec2 wave = shotgun(3700.0 * t, 2.4, 0.0, 1.0);
    dest += 0.1 * duck * env * tanh(8.0 * wave);
  }

  { // open hihat
    vec4 seq = seq16(time.y, 0x2626);
    float t = seq.t;
    float q = seq.q;

    float env = exp2(-40.0 * t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      float odd = float(i % 2);
      float tt = (t + 0.3) * mix(1.0, 1.002, odd);
      vec3 dice = hash3f(vec3(i / 2));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(11.30 + 0.9 * dice.x) * tt + dice2.xy);
      wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.78 + 0.1 * dice.y) * tt + dice2.yz);
      wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.92 + 0.8 * dice.z) * tt + dice2.zx);

      sum += wave * mix(1.0, 0.5, odd);
    }

    dest += 0.18 * env * duck * tanh(sum);
  }

  // { // ride
  //   vec4 seq = seq16(time.y, 0xaaaa);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = exp2(-8.0 * t);

  //   vec2 wave = shotgun(5000.0 * t, 1.1, 0.0, 1.0);
  //   dest += 0.08 * mix(0.1, 1.0, duck) * env * tanh(8.0 * wave);
  // }

  { // clap
    vec4 seq = seq16(time.y, 0x0001);
    float t = seq.y;
    float q = seq.w;

    float env = mix(
      exp2(-60.0 * t),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-100.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(1400.0 * t), 1840.0 * t), 0.5, 2.0).xy;

    dest += 0.12 * tanh(20.0 * env * wave);
  }

  { // rim
    vec4 seq = seq16(time.y, 0x5151);
    float t = seq.y;

    float env = exp2(-400.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * env * vec2(wave) * rotate2D(seq.x);
  }

  // { // clav
  //   vec4 seq = seq16(time.y, 0x8080);
  //   float t = seq.y;

  //   float env = mix(
  //     exp2(-100.0 * t),
  //     exp2(-1.0 * t),
  //     0.004
  //   );

  //   vec2 wave = tri(5000.0 * t + vec2(0.0, 0.25));

  //   dest += 0.3 * env * vec2(wave);
  // }

  { // fm shot
    float t = tmod(time - 4.5 * B2T, 16.0 * B2T);

    float env = smoothstep(0.0, 0.001, t) * exp(-4.0 * t);

    vec2 sum = vec2(0.0);
    repeat(i, 32) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(i / 2, 11, 12));
      float pol = u2b(float(i % 2));
      vec2 dicen = boxMuller(dice.xy);

      float freq = 100.0;
      freq *= exp2(0.02 * dicen.x);
      float phase = freq * (t + 0.0003 * pol) + dice.z;
      phase += 0.5 * exp(-0.5 * t) * sin(TAU * 0.42 * phase + sin(TAU * 1.88 * phase));

      float wave = pol * sin(TAU * phase);
      sum += vec2(wave) * rotate2D(TAU * dice.y);
    }

    dest += 0.12 * mix(0.2, 1.0, duck) * env * sum;
  }

  { // crash
    float t = time.z;

    float env = mix(
      exp2(-4.0 * t),
      exp2(-1.0 * t),
      0.2
    );

    vec2 wave = shotgun(3500.0 * t, 2.8, 0.0, 0.3);
    dest += 0.17 * mix(0.2, 1.0, duck) * env * tanh(8.0 * wave);
  }

  { // beep
    vec2 sum = vec2(0.0);
    repeat(i, 3) {
      const float N_STEP = 8.0;
      float s = float[](2.0, 5.0, 7.0)[i];
      float pitchoff = float[](-0.4, -4.6, 9.5)[i];

      repeat(iDelay, 5) {
        float fiDelay = float(iDelay);
        float delaydecay = exp(-fiDelay);
        float t = tmod(time - s * S2T - 2.0 * S2T * fiDelay, N_STEP * S2T);

        float env = smoothstep(0.0, 0.001, t);
        env *= mix(
          smoothstep(0.0, 0.001, 0.09 - t),
          exp2(-2.0 * t),
          0.03
        );

        float pitch = 60.0 + TRANSPOSE + pitchoff;
        float freq = p2f(pitch);
        vec2 phase = vec2(freq * t);
        phase += vec2(0.0, 0.25);
        phase += 0.04 * cheapnoise(64.0 * t);
        vec2 wave = cheapFilterSaw(phase, 0.9);
        sum += delaydecay * env * wave * rotate2D(time.w);
      }
    }
    dest += 0.12 * mix(0.2, 1.0, duck) * sum;
  }

  { // bell
    vec2 sum = vec2(0.0);
    repeat(iDelay, 5) {
      float fiDelay = float(iDelay);
      float delaydecay = exp(-fiDelay);
      float toff = -2.1 * S2T * fiDelay;

      vec4 seq = seq16(time.y + toff, 0xffff);
      float s = seq.s + 16.0 * mod(floor((time.z + toff) / timeLength.y), 16.0);
      float t = seq.t;
      float q = seq.q;

      if (mod(s, 64.0) < 57.0) { continue; }

      vec3 dice = hash3f(vec3(s, 7, 112));

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      env *= exp(-20.0 * t);

      float freq = exp2(7.0 + 3.0 * dice.x);
      float phase = freq * t + dice.z;
      float fm = mix(
        0.5 * sin(1.1 * TAU * phase),
        8.0 * sin(0.546 * TAU * phase),
        paramFetch(param_knob4)
      );
      vec2 wave = cis(TAU * phase + fm);

      sum += delaydecay * env * wave;
    }

    dest += 0.3 * mix(0.2, 1.0, duck) * sum;
  }

  // { // chord
  //   int CHORD[] = int[](
  //     0, 2, 3, 5, 7, 10, 12, 14
  //   );
  //   vec2 sum=vec2(0);

  //   float t = time.z;

  //   int N_OSC = 64;
  //   repeat(i, N_OSC) {
  //     float fi = float(i);
  //     float iprog = fi / float(N_OSC);
  //     vec3 dice = hash3f(vec3(i, 21, 28));

  //     float t = tmod(time - 16.0 * B2T * iprog, 16.0 * B2T);
  //     float tprog = t / (16.0 * B2T);
  //     float fade = sin(PI * tprog);

  //     float pitch = 60.0 + TRANSPOSE + float(CHORD[i % 8]);
  //     pitch += 0.08 * boxMuller(dice.xy).x;
  //     float freq = p2f(pitch);

  //     vec3 p = vec3(0.0, 0.0, 40.0);
  //     p.xy += 1.0 * cis(TAU * freq * t);
  //     p.yz += 0.2 * cis(time.w);
  //     vec3 p2 = p + vec3(0.0, 0.0, 0.2);
  //     vec2 wave = cyclic(p, 0.5, 1.6).xy;
  //     wave -= cyclic(p2, 0.5, 1.6).xy;
  //     sum += 0.2 * fade * wave * rotate2D(fi);
  //   }

  //   dest += 1.0*0.3 * mix(0.4, 1.0, duck) * sum;
  // }

  return clip(1.3 * tanh(dest));
}
