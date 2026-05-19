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
#define TRANSPOSE -2.5

const float SWING = 0.5;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

uniform vec4 param_knob0; // glitch
uniform vec4 param_knob2; // snareroll
uniform vec4 param_knob3; // kick cut
uniform vec4 param_knob4; // acid cutoff

#define p0 paramFetch(param_knob0)
#define p2 paramFetch(param_knob2)
#define p3 paramFetch(param_knob3)
#define p4 paramFetch(param_knob4)

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

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

vec3 cyclic(vec3 p, float pers, float lacu, float warp) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  repeat(i, 5) {
    p *= rot;
    p += warp * sin(p.zxy);
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

vec4 glitchTime(vec4 time) {
  float s = 2.0 * floor(time.z / (2.0 * S2T));
  float t = mod(time.x, 2.0 * S2T);
  vec3 dice = hash3f(vec3(s, 18, 10));

  if (dice.x < 0.1) {
    time -= 2.0 * S2T; // prev 2 step
  } else if (dice.x < 0.3) {
    time -= lofi(t, S2T); // retrigger 1 step
  } else if (dice.x < 0.4) {
    time -= lofi(t, 0.5 * S2T); // retrigger 0.5 step
  } else if (dice.x < 0.5) {
    time -= lofi(t, 0.25 * S2T); // retrigger 0.25 step
  } else if (dice.x < 0.6) {
    time -= 0.8 * lofi(t, 0.4 * S2T); // stretch 0.2x 0.4 step
  } else if (dice.x < 0.7) {
    time -= 0.7 * lofi(t, 0.2 * S2T); // stretch 0.3x 0.2 step
  } else if (dice.x < 0.8) {
    time -= 0.5 * t; // slow 0.5x
  } else if (dice.x < 0.9) {
    time += (1.0 - exp(-5.0 * t)) / 5.0 - t; // stop
  }

  return mod(time, timeLength);
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float duck = 1.0;

  float enableProgression = 0.0;
  // { // enable progression
  //   enableProgression = 1.0;
  // }
  #define GET_TRANS(t) float[](0.0, 0.0, 3.0, -3.0)[int(enableProgression * t / (8.0 * B2T)) % 4]

  bool isFillIn = false;
  // { // enable fill-in
  //   isFillIn = time.z > 60.0 * B2T;
  // }

  { // glitch
    time = mix(
      time,
      glitchTime(time),
      step(0.5, param_knob0.x)
    );
  }

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;

    if (isFillIn) {
      seq = seq16(time.y, 0x8daa);
      t = seq.t;
      q = seq.q;
    }

    duck = min(
      duck,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    {
      float env = smoothstep(0.0, 0.001, q);
      env *= smoothstep(0.3, 0.1, t);
      env *= mix(1.0, exp2(-80.0 * t), p3); // hpf-like

      vec2 phase = (
        50.0 * t
        - 10.0 * exp2(-t * 30.0)
        - 2.0 * exp2(-t * 400.0)
        + 0.1 * exp2(-40.0 * t) * cheapnoise(t)
      );

      vec2 wave = tanh(2.0 * sin(TAU * phase));

      dest += 0.7 * env * wave;
    }
  }

  { // toms
    vec4 seqh = seq16(time.y, 0x1010);
    vec4 seql = seq16(time.y, 0x0404);
    float t = min(seqh.t, seql.t);
    float q = min(seqh.q, seql.q);
    float isHi = step(seqh.t, seql.t);

    float env = exp(-20.0 * t);
    float freq = mix(80.0, 110.0, isHi);
    float phase = (
      t
      - 0.03 * exp2(-40.0 * t)
      - 0.01 * exp2(-150.0 * t)
    );
    phase *= TAU * freq;

    float fm = 3.0 * exp2(-10.0 * t) * sin(1.3 * phase);
    vec2 wave = cis(phase + fm + 10.0 * t);
    wave *= mix(vec2(0.5, 1.0), vec2(1.0, 0.5), isHi);

    dest += 0.2 * mix(0.8, 1.0, duck) * tanh(2.0 * env * wave);
  }

  // { // perc
  //   vec4 seq = seq16(time.y, 0xabda);
  //   float t = seq.t;
  //   float q = seq.q;

  //   if (isFillIn) {
  //     seq = seq16(time.y, 0x6dff);
  //     t = seq.t;
  //     q = seq.q;

  //     if (time.y > 3.0 * B2T) {
  //       float l = 0.5 * S2T;
  //       t = mod(t, l);
  //       q = l - t;
  //     }
  //   }

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-100.0 * t);

  //   vec3 p = vec3(0.0);
  //   p.xy += cis(3000.0 * t);
  //   p.z += 600.0 * t;
  //   vec2 wave = cyclic(p, 0.5, 2.0, 0.3).xy;
  //   wave = tanh(4.0 * wave);

  //   dest += 0.4 * mix(0.1, 1.0, duck) * env * wave;
  // }

  // { // hihat
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp2(-150.0 * t);

  //   vec2 wave = shotgun(1600.0 * t, 3.0, 0.0, 1.0);
  //   wave = tanh(20.0 * wave);

  //   dest += 0.2 * mix(0.1, 1.0, duck) * env * wave;
  // }

  // { // clap
  //   vec4 seq = seq16(time.y, 0x0808);
  //   float t = seq.t;
  //   float q = seq.q;

  //   if (isFillIn) {
  //     seq = seq16(time.y, 0x2daa);
  //     t = seq.t;
  //     q = seq.q;
  //   }

  //   float env = mix(
  //     exp2(-80.0 * t),
  //     exp2(-500.0 * mod(t, 0.012)),
  //     exp2(-100.0 * max(0.0, t - 0.02))
  //   );

  //   vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0, 1.0).xy;

  //   dest += 0.4 * tanh(20.0 * env * wave);
  // }

  // { // cowbell
  //   vec4 seq = seq16(time.y, 0x0100);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = exp2(-t * 30.0);

  //   vec2 phase = t * vec2(450.0, 460.0);

  //   vec2 wave = tri(phase) + tri(1.48 * phase) + tri(2.21 * phase);

  //   dest += 0.16 * mix(0.1, 1.0, duck) * tanh(8.0 * env * wave);
  // }

  // { // open hihat
  //   vec4 seq = seq16(time.y, 0x2222);
  //   float t = seq.y;

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 6.0 * exp2(-3.0 * t) * sin(wave + exp2(13.10 + 0.1 * dice.x) * t + dice2.xy);
  //     wave = 3.0 * exp2(-4.0 * t) * sin(wave + exp2(11.28 + 0.1 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * exp2(-20.0 * t) * sin(wave + exp2(13.12 + 0.2 * dice.z) * t + dice2.zx);

  //     sum += wave;
  //   }

  //   dest += 0.15 * mix(0.0, 1.0, duck) * tanh(2.0 * sum);
  // }

  // { // ride
  //   vec4 seq = seq16(time.y, 0xaaaa);
  //   float st = seq.s;
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp(-10.0 * t);

  //   vec2 wave = shotgun(3000.0 * t, 3.5, 0.0, 1.0);
  //   wave = tanh(4.0 * wave);

  //   dest += 0.4 * mix(0.1, 1.0, duck) * env * wave;
  // }

  { // snare roll
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float amp = p2 * smoothstep(32.0 * B2T, 64.0 * B2T, time.z);
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

    dest += 0.24 * amp * mix(0.5, 1.0, duck) * tanh(8.0 * env * wave);
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

  { // acid
    vec4 seq = seq16(time.y, 0xd2d2);
    float s = seq.s;
    float t = seq.t;
    float q = seq.q;
    q -= S2T * 0.2;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q);
    env *= exp2(-4.0 * t);

    float cutoff = (
      8.0
      + 2.0 * p4
      + mix(2.0, 5.0, mod(seq.s, 2.0)) * (smoothstep(0.0, 0.01, t) * exp2(-7.0 * t) - 0.5)
    );
    float cfreq = exp2(cutoff);
    float reso = 0.8;

    int i = 0;
    float pitch = 36.0 + TRANSPOSE + GET_TRANS(time.z);
    float basefreq = p2f(pitch);
    float basephase = t * basefreq;

    vec2 sum = vec2(0.0);

    repeat(i, 128) { // acid
      float fi = float(i);
      vec3 dice = hash3f(vec3(i, 5, 7));

      float p = 1.0 + 2.0 * fi;
      float freq = basefreq * p;
      float coeff = exp(-0.1 * p);

      vec2 filt = ladderLPF(freq, cfreq, reso);
      float phase = basephase * p;

      vec2 wave = vec2(0.0);
      wave += sin(TAU * phase + filt.y);
      wave *= rotate2D(0.2 * p * (dice.y - 0.5));
      sum += wave * env * coeff * filt.x;
    }

    float bias = -0.07;
    sum = clip(4.0 * (sum + bias)) - bias;

    { // sub
      float pitch = 24.0 + TRANSPOSE;
      float freq = p2f(pitch);
      float phase = t * freq;

      float wave = mix(
        sin(TAU * phase),
        sin(TAU * 2.0 * phase),
        0.4
      );
      wave = tanh(1.5 * wave);

      sum += env * wave;
    }

    dest += 0.3 * mix(0.2, 1.0, duck) * sum;
  }

  // { // chord
  //   const int N_NOTES = 6;
  //   const int NOTES[] = int[](
  //     0, 3, 7, 10, 14, 19
  //   );

  //   vec4 seqg = seq16(time.y, 0xffff);

  //   vec2 sum = vec2(0.0);
  //   repeat(iUnison, 24) {
  //     vec3 dice = hash3f(vec3(iUnison, 7, 7));
  //     vec2 dicen = boxMuller(dice.xy);

  //     vec4 tdelay = mod(time - 0.1 * dice.y, timeLength);
  //     float l = 8.0 * B2T;
  //     float t = mod(tdelay.z, l);
  //     float st = round((tdelay.z - t) / S2T);

  //     t += 0.001 * sin(TAU * (1.0 * t + dice.x)); // chorus

  //     float fade = smoothstep(0.0, S2T, t) * smoothstep(0.0, S2T, l - t);

  //     int iNote = iUnison % N_NOTES;
  //     float pitch = 48.0 + TRANSPOSE + GET_TRANS(tdelay.z);
  //     pitch += float(NOTES[iNote]);
  //     float freq = p2f(pitch);

  //     float phase = t * freq + dice.y;
  //     phase = lofi(phase, 1.0 / 32.0);

  //     vec2 wave = vec2(
  //       cheapfiltersaw(phase, 1.0) - sin(TAU * phase + PI)
  //     );

  //     sum += fade * tanh(wave) * rotate2D(2.4 * float(iUnison));
  //   }

  //   dest += 0.07 * sum * mix(0.2, 1.0, duck);
  // }

  return clip(1.2 * tanh(dest));
}
