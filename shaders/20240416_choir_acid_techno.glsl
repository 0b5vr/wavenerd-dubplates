#define TRANSPOSE -3.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float LN2 = log(2.0);
const float PI = acos(-1.0);
const float TAU = PI * 2.0;
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

vec4 seq16(float t, float tps, int seq) {
  int sti = int(t / tps) & 15;
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - prevStepBehind;
  float nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + nextStepForward;

  return vec4(
    prevStep,
    mod(t, 16.0 * tps) - prevStep * tps,
    nextStep,
    nextStep * tps - mod(t, 16.0 * tps)
  );
}

vec4 quant(float x, float ks, float kt, out float i) {
  i = floor(floor(x / ks + 1E-4) * ks / kt + 1E-4);

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

vec4 quant(float x, float ks, float kt) {
  float i;
  return quant(x, ks, kt, i);
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
  float sidechain = smoothstep(0.0, 0.4, time.x) * smoothstep(0.0, 0.001, B2T - time.x);

  { // kick
    int pattern = time.z / B2T < 60.0 ? 0x8888 : 0x8064;
    vec4 seq = seq16(mod(time.z, 32.0 * S2T), S2T, pattern);
    float t = seq.t;
    float q = seq.q;
    sidechain = min(
      sidechain,
      smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q)
    );

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.2, 0.1, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    float tt = t;
    float wave = sin(
      250.0 * tt
      - 40.0 * exp(-tt * 10.0)
      - 30.0 * exp(-tt * 40.0)
      - 20.0 * exp(-tt * 200.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  { // sub bass
    int NOTES[16] = int[](
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 12,
      5, 0, 7, 12
    );

    vec4 seq = seq16(time.y, S2T, 0x889b);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.02, q);

    float pitch = 36.0 + TRANSPOSE + float(NOTES[int(seq.x)]);
    float freq = p2f(pitch);
    float phase = freq * t - exp(-t * 100.0);

    float wave = (
      sin(TAU * phase)
      + 0.2 * sin(TAU * phase * 3.0)
    );
    dest += 0.6 * sidechain * tanh(env * wave);
  }

  { // hihat
    float t = mod(time.z, 64.0 * S2T);
    float st = lofi(t, S2T);
    t -= st;
    float q = st - t;

    float k = exp2(6.0 - 3.0 * fract(0.9 - 0.632 * st / S2T));
    float env = exp(-k * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.1 * env * mix(0.5, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // open hihat
    float t = mod(time.x - 2.0 * S2T, B2T);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * exp(-4.0 * t) * sin(wave + exp2(12.40 + 0.3 * dice.x) * t + dice2.xy);
      wave = 3.2 * exp(-3.0 * t) * sin(wave + exp2(11.57 + 0.3 * dice.y) * t + dice2.yz);
      wave = 1.0 * exp(-4.0 * t) * sin(wave + exp2(15.18 + 0.3 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.2 * exp(-8.0 * t) * sidechain * tanh(sum);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.41 + 0.63);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);
    vec2 wave = cyclic(vec3(cis(200.0 * t), exp2(9.0 + 2.0 * vel) * t), 1.0, 2.0).xy;
    dest += 0.2 * env * sidechain * tanh(2.0 * wave);
  }

  { // clap
    vec4 seq = seq16(time.y, S2T, 0x0808);
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      exp(-26.0 * t),
      exp(-200.0 * mod(t, 0.013)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(900.0 * t), 1940.0 * t), 0.7, 2.0).xy;

    dest += 0.3 * tanh(8.0 * env * wave);
  }

  { // ride
    float t = time.x;

    float env = exp(-2.0 * t);

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

    dest += 0.06 * env * sidechain * tanh(sum);
  }

  if (60.0 * B2T < time.z) { // snare909
    float t = mix(
      mod(time.x, S2T),
      mod(time.x, S2T * 0.5),
      step(2.0, time.y / B2T)
    );

    float env = exp(-20.0 * t);
    float fade = mix(0.5, 1.0, smoothstep(0.0, 4.0, time.y / B2T))
      * smoothstep(4.0, 3.8, time.y / B2T);

    vec2 wave = (
      cheapnoise(64.0 * t)
      + sin(1400.0 * t - 80.0 * exp(-t * 120.0))
    );

    dest += 0.2 * fade * tanh(3.0 * env * wave);
  }

  { // crash
    float t = mod(time.z, 60.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 3.0);
    dest += 0.4 * env * mix(0.3, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // acid
    const int N_NOTES = 5;
    const int NOTES[N_NOTES] = int[](0, 12, 0, 24, 13);

    float seqi;
    vec4 seq = quant(time.z / S2T, 1.0, 1.12, seqi) * S2T;
    float t = seq.t;
    float q = seq.q;

    vec2 sum = vec2(0.0);

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q - 0.02);

    float cutoff = exp2(8.0 + 2.0 * fract(seqi * 0.612) + 3.0 * smoothstep(0.0, 0.01, t) * exp(-8.0 * t));
    float reso = 0.8;

    repeat(i, 128) {
      float fi = float(i);

      float pitch = 36.0 + TRANSPOSE + float(NOTES[int(seqi) % N_NOTES]);

      float p = 1.0 + fi;
      float freq = p2f(pitch) * p;

      vec2 filt = ladderLPF(freq, cutoff, reso);
      float phase = t * freq;

      vec2 wave = vec2(0.0);
      wave += cis(TAU * phase + filt.y);
      sum += wave * env * filt.x / p * rotate2D(2.4 * fi + time.w);
    }

    dest += 0.3 * tanh(2.0 * sum);
  }

  { // arp
    const int N_NOTES = 8;
    const int NOTES[N_NOTES] = int[](0, 0, 7, 7, 12, 15, 17, 22);

    vec2 sum = vec2(0.0);

    repeat(i, 5) {
      float fi = float(i);

      float seqi;
      vec4 seq = quant(time.z / S2T - 3.0 * fi, 1.0, 1.0, seqi) * S2T;
      float t = seq.t;
      float q = seq.q;

      vec3 dice = hash3f(vec3(i, seqi, 3));
      vec2 dicen = boxMuller(dice.xy);

      float env = (
        smoothstep(0.0, 0.001, t)
        * smoothstep(0.0, 0.01, q)
        * exp(-t * 7.0)
      );

      float pitch = 60.0 + TRANSPOSE + float(NOTES[int(seqi * 3.31) % N_NOTES]);
      float freq = p2f(pitch);

      vec2 phase = t * freq + dice.x + vec2(0.0, 0.25);
      phase *= exp2(0.002 * dicen);

      vec2 wave = (
        (fract(phase) * 2.0 - 1.0)
        - (fract(P5 * phase) * 2.0 - 1.0)
      );

      sum += env * wave * exp(-fi);
    }

    dest += 0.2 * mix(0.1, 1.0, sidechain) * sum;
  }

  { // chord
    const int N_CHORD = 8;
    int CHORD[N_CHORD * 2] = int[](
      -12, 0, 7, 10, 14, 17, 24, 29,
      -11, 1, 8, 10, 15, 17, 24, 31
    );

    vec2 sum = vec2(0.0);

    vec4 seqg = quant(mod(time.z / S2T, 64.0), 1.0, 1.28);
    float gate = (
      smoothstep(0.0, 0.001, seqg.y)
      * smoothstep(0.0, 0.001, seqg.w)
      * exp(-6.0 * max(0.4 - seqg.w, 0.0))
    );

    repeat(iUnison, 128) {
      int progress = iUnison / 64;

      float t = mod(time.z + (8.0 * float(progress) + 0.5) * B2T, 16.0 * B2T);
      float fade = smoothstep(0.0, 4.0, t / S2T) * smoothstep(-4.0, 0.0, 32.0 - t / S2T);

      vec3 dice = hash3f(vec3(iUnison, 7, 7));
      vec2 dicen = boxMuller(dice.xy);

      float pitch = 48.0 + TRANSPOSE + float(CHORD[iUnison % N_CHORD + N_CHORD * progress]);
      float freq = p2f(pitch);

      float phase = t * freq + dice.x;
      phase *= exp(0.01 * dicen.x);
      phase += 0.0001 * freq * sin(30.0 * t + dice.x);

      float pers = 0.4;
      pers *= 0.01 + fade;

      vec3 c = vec3(0.0);
      vec3 d = vec3(2.0, -3.0, -8.0);

      vec2 wave = mix(
        cyclic(fract(phase) * d, pers, 2.0),
        cyclic((fract(phase) - 1.0) * d, pers, 2.0),
        smoothstep(0.9, 1.0, fract(phase))
      ).xy;

      float amp = 0.1 * fade * smoothstep(0.0, freq / p2f(36.0 + TRANSPOSE), t / S2T);
      sum += amp * wave * rotate2D(float(iUnison));
    }

    dest += sum * mix(0.0, 1.0, gate) * mix(0.2, 1.0, sidechain);
    // dest *= mix(0.0, 1.0, gate);
  }

  return tanh(dest);
}
