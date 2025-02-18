// { LFG }
// #define LFG

#define TRANSPOSE -1.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(i) clamp(i, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

const float SWING = 0.5;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);
const float MIN3 = pow(2.0, 3.0 / 12.0);
const float P4 = pow(2.0, 5.0 / 12.0);
const float P5 = pow(2.0, 7.0 / 12.0);

uniform vec4 param_knob0; // pad amp
uniform vec4 param_knob1; // clap rhythm
uniform vec4 param_knob3; // kick fill-in at 16th bar
uniform vec4 param_knob7; // kick cut

#define p0 paramFetch(param_knob0)
#define p1 paramFetch(param_knob1)
#define p3 paramFetch(param_knob3)
#define p7 paramFetch(param_knob7)

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

  for (int i = 0; i ++ < 5;) {
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
  float duck = 1.0;

  int N_CHORD = 8;
  int CHORDS[] = int[](
    -4, 3, 10, 12, 14, 15, 17, 19,
    -5, 2, 7, 9, 10, 12, 14, 17
  );

  bool fillIn = false;
  #ifdef LFG
    fillIn = time.z > 63.5 * B2T;
  #endif

  // { // kick
  //   float t = time.x;
  //   float q = B2T - t;

  //   if (fillIn) {
  //     t = mod(t, 0.5 * B2T);
  //     q = 0.5 * B2T - t;
  //   }

  //   duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);

  //   float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);
  //   #ifdef LFG
  //     env *= fillIn ? 1.0 : exp(-70.0 * t);
  //   #else
  //     env *= mix(1.0, exp(-70.0 * t), p7);
  //   #endif

  //   {
  //     float wave = sin(TAU * (
  //       40.0 * t
  //       - 4.0 * exp2(-t * 30.0)
  //       - 3.0 * exp2(-t * 90.0)
  //       - 4.0 * exp2(-t * 500.0)
  //     ));
  //     dest += 0.6 * tanh(2.0 * env * wave);
  //   }

  //   if (fillIn) {
  //     return dest;
  //   }
  // }

  // { // bass
  //   float t = time.x;
  //   float q = B2T - t;

  //   float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

  //   int chordhead = N_CHORD * int(mod(time.z / (16.0 * B2T), 2.0));
  //   float pitch = 36.0 + TRANSPOSE + float(CHORDS[chordhead]);
  //   float freq = p2f(pitch);
  //   float phase = freq * t;

  //   float wave = tanh(sin(TAU * phase));
  //   dest += 0.5 * env * mix(0.0, 1.0, duck) * wave;
  // }

  // { // hihat
  //   float st = mod(floor(time.z / S2T), 64.0);
  //   vec3 dice = hash3f(vec3(st, 3, 5));

  //   float l = exp2(-floor(3.0 * pow(dice.x, 10.0))) * S2T;
  //   float t = mod(time.x, l);

  //   float vel = fract(st * 0.38);
  //   float env = exp2(-exp2(6.0 - 1.0 * vel) * t);

  //   vec2 wave = mix(
  //     fract((sin(((t + vec2(0.0, 2.0)) * 4444.141)) * 15.56)) - 0.5,
  //     shotgun(6000.0 * t, 2.0, 0.0, 1.0),
  //     0.5
  //   );

  //   dest += 0.12 * env * mix(0.2, 1.0, duck) * tanh(8.0 * wave);
  // }

  // { // open hihat
  //   float t = mod(time.x - 0.5 * B2T, B2T);
  //   float q = B2T - t;

  //   float env = exp2(-14.0 * t) * smoothstep(0.0, 0.01, q);

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

  //   dest += 0.18 * env * duck * tanh(sum);
  // }

  // { // shaker
  //   float t = mod(time.x, S2T);
  //   float st = mod(floor(time.y / S2T), 8.0);

  //   float vel = fract(st * 0.58 + 0.13);
  //   float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);

  //   float phase = 240.0 * t;
  //   phase += phase + 0.1 * sin(TAU * phase);
  //   vec2 wave = shotgun(phase, 2.0, 0.4, exp2(mix(1.0, 3.0, vel)));

  //   dest += 0.16 * env * duck * tanh(8.0 * wave);
  // }

  // { // clap
  //   float l = 2.0 * B2T;
  //   #ifdef LFG
  //     l = B2T;
  //   #endif

  //   float t = tmod(time - B2T, l);
  //   float q = l - t;

  //   float env = mix(
  //     exp2(-80.0 * t),
  //     exp2(-500.0 * mod(t, 0.012)),
  //     exp2(-100.0 * max(0.0, t - 0.02))
  //   );

  //   vec2 wave = cyclic(vec3(4.0 * cis(800.0 * t), 840.0 * t), 0.5, 2.0).xy;

  //   dest += 0.2 * tanh(20.0 * env * wave);
  // }

  // { // rim
  //   float t = mod(
  //     mod(
  //       mod(time.z, 2.75 * B2T),
  //       1.25 * B2T
  //     ),
  //     0.5 * B2T
  //   );
  //   float st = round((time.z - t) / S2T);

  //   float env = step(0.0, t) * exp2(-400.0 * t);

  //   float wave = tanh(4.0 * (
  //     + tri(t * 400.0 - 0.5 * env)
  //     + tri(t * 1500.0 - 0.5 * env)
  //   ));

  //   dest += 0.2 * env * vec2(wave) * rotate2D(st);
  // }

  // { // fm perc
  //   float l = S2T;
  //   float t = mod(time.x, l);
  //   float q = l - t;
  //   float st = round((time.z - t) / S2T);
  //   vec3 dice = hash3f(vec3(mod(st, 64.0), 4, 8));

  //   float freq = exp2(8.0 + 2.0 * dice.x);
  //   float env = exp2(-exp2(3.0 + 3.0 * dice.y) * t) * smoothstep(0.0, 0.01, q);
  //   float fm = env * exp2(6.0 + 1.0 * dice.z) * sin(freq * exp2(-t));
  //   float wave = sin(fm);
  //   dest += 0.05 * duck * vec2(wave) * rotate2D(st);
  // }

  // { // ride
  //   float l = 0.5 * B2T;
  //   float t = tmod(time, l);
  //   float q = l - t;

  //   float env = exp2(-4.0 * t) * smoothstep(0.0, 0.01, q);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i, 7, 4));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 2.9 * exp(-3.0 * t) * sin(wave + exp2(13.10 + 0.8 * dice.x) * t + dice2.xy);
  //     wave = 2.8 * exp(-1.0 * t) * sin(wave + exp2(14.57 + 0.1 * dice.y) * t + dice2.yz);
  //     wave = 1.0 * sin(wave + exp2(14.09 + 1.0 * dice.z) * t + dice2.zx);

  //     sum += wave / 4.0;
  //   }

  //   dest += 0.05 * env * mix(0.3, 1.0, duck) * tanh(2.0 * sum);
  // }

  #ifdef LFG
    { // snare909
      float fade = smoothstep(32.0 * B2T, 64.0 * B2T, time.z);

      float t = mod(time.x, S2T);
      float q = S2T - t;

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

      float sinphase = 214.0 * t - 4.0 * exp2(-t * 200.0);
      float noisephase = 128.0 * t;
      vec2 wave = mix(
        mix(
          cis(TAU * (sinphase)),
          cis(TAU * (1.5 * sinphase)),
          0.3
        ),
        cheapnoise(noisephase) - cheapnoise(noisephase - 0.004),
        0.4
      );

      dest += 0.14 * fade * mix(p0, 1.0, duck) * tanh(4.0 * env * wave);
    }

    if (time.z > 48.0 * B2T) { // sweep
      float t = tmod(time, 16.0 * B2T);

      float env = smoothstep(0.0, 16.0 * B2T, t);

      vec2 wave = vec2(0.0);
      wave += cheapnoise(128.0 * t);
      wave += cheapnoise(128.0 * (t + 0.002 * exp(-0.4 * t)));
      wave += cheapnoise(128.0 * (t + 0.004 * exp(-0.4 * t)));

      dest += 0.07 * env * wave;
    }
  #endif

  // { // crash
  //   float t = mod(time.z, 64.0 * B2T);

  //   float env = mix(exp(-t), exp(-5.0 * t), 0.7);
  //   vec2 wave = shotgun(3800.0 * t, 1.5, 0.0, 3.0);
  //   dest += 0.3 * mix(0.2, 1.0, duck) * env * tanh(8.0 * wave);
  // }

  { // choir
    float l = 16.0 * B2T;
    float t = tmod(time, l);
    float q = l - t;
    int chordhead = N_CHORD * int(mod(time.z / l, 2.0));

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);
    repeat(i, 64) {
      float fi = float(i);
      vec3 dice = hash3f(float(i) + vec3(8, 4, 2));

      float note = 48.0 + TRANSPOSE + float(CHORDS[(i % N_CHORD) + chordhead]);
      float freq = p2f(note) * exp2(0.016 * tan(2.0 * dice.y - 1.0));
      float phase = t * freq;
      phase += 0.04 * sin(2.0 * TAU * phase);

      vec3 d = vec3(2.0, -3.0, -8.0);
      vec2 wave = (
        cyclic(fract(phase) * d, 0.5, 1.7).xy
        - cyclic(fract(phase + 0.08) * d, 0.5, 1.7).xy
      );

      sum += vec2(wave) * rotate2D(fi);
    }

    dest += p0 * 0.03 * mix(0.2, 1.0, duck) * env * sum;
  }

  { // sinearp
    int ARP[] = int[](
      0, -5, 3, -2, 0, 12, 0, 7,
      -5, 3, -2, 12, -5, 7, 3, -2
    );

    vec2 sum = vec2(0.0);
    repeat(iDelay, 3) {
      float fiDelay = float(iDelay);
      float delaydecay = exp(-1.0 * fiDelay);

      float l = 0.5 * B2T;
      float t = tmod(time - l * fiDelay, 8.0 * B2T);
      float st = mod(floor(t / l), 16.0);
      t = mod(time.x, l);
      float q = l - t;

      vec3 seed = hash3f(vec3(st, 5, 8));

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

      float pitch = 72.0 + TRANSPOSE + float(ARP[int(st)]);
      float freq = p2f(pitch);
      float phase = t * freq;

      vec2 wave = vec2(
        + cis(2.0 * TAU * (phase + seed.x))
        + cis(1.5 * TAU * (phase + seed.y))
      );

      sum += delaydecay * env * wave;
    }
    dest += 0.09 * mix(0.3, 1.0, duck) * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  return clip(1.3 * tanh(dest));
}
