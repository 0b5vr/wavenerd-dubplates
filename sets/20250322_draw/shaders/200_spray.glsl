#define TRANSPOSE 0.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(i) clamp(i, -1., 1.)
#define lofi(i,m) (floor((i) / (m)) * (m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define tri(p) (1.-4.*abs(fract(p-0.25)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

const float SWING = 0.64; // -> 0.52

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);
const float MIN3 = pow(2.0, 3.0 / 12.0);
const float P4 = pow(2.0, 5.0 / 12.0);
const float P5 = pow(2.0, 7.0 / 12.0);

uniform vec4 param_knob7; // kick cut

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

vec2 spray(float t, float freq, float spread, float seed, float interval, int count) {
  float grainLength = float(count) * interval;

  vec2 sum = vec2(0.0);
  repeat(i, count) {
    float fi = float(i);

    float off = -interval * fi;
    float tg = mod(t + off, grainLength);
    float prog = tg / grainLength;

    vec3 dice = hash3f(vec3(i, lofi(t + off, grainLength), seed));
    vec2 dicen = boxMuller(dice.xy);

    float envg = smoothstep(0.0, 0.5, prog) * smoothstep(1.0, 0.5, prog);

    vec2 phase = vec2(freq * t);
    phase *= exp2(spread * dicen.xy);
    phase += dice.xy;

    vec2 wave = sin(TAU * phase);
    sum += 2.0 * envg * wave;
  }

  return sum / float(count);
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float duck = 1.0;

  { // kick
    vec4 seq = seq16(time.y, 0x9294);
    float t = seq.t;
    float q = seq.q;
    duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);
    env *= mix(1.0, exp2(-40.0 * t), p7);

    // {
    //   float wave = sin(TAU * (
    //     45.0 * t
    //     - 1.0 * exp2(-t * 20.0)
    //     - mix(4.0, 5.0, mod(seq.s, 8.0) != 3.0) * exp2(-t * 60.0)
    //   ));
    //   dest += 0.6 * clip(2.0 * tri(0.40 * env * wave));
    // }
  }

  // { // sub kick
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

  //   {
  //     float wave = sin(TAU * (
  //       42.0 * t
  //       - 4.0 * exp2(-t * 40.0)
  //     ));
  //     dest += 0.4 * duck * tanh(2.0 * env * wave);
  //   }
  // }

  // { // rumble
  //   vec4 seq = seq16(time.y, 0x8888);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

  //   vec2 wave = spray(t, 90.0, 0.5, 0.0, 0.01, 4);
  //   wave = mix(wave, vec2(dot(wave, vec2(0.5))), 0.5);
  //   dest += 0.2 * duck * env * tanh(3.0 * wave);
  // }

  // { // hihat
  //   float st;
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = exp(-exp2(5.0) * t) * smoothstep(0.0, 0.01, q);
  //   vec2 wave = spray(t, 8000.0, 0.4, 0.0, 0.001, 64);
  //   dest += 0.15 * env * mix(0.3, 1.0, duck) * tanh(8.0 * wave);
  // }

  // { // clap
  //   float st;
  //   vec4 seq = seq16(time.y, 0x2346);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.01, q);
  //   env *= exp(-30.0 * t);
  //   vec2 wave = spray(t, 1400.0, 0.5, 4.0, 0.002, 8);
  //   dest += 0.4 * env * mix(0.7, 1.0, duck) * tanh(4.0 * wave);
  // }

  // { // ride
  //   float st;
  //   vec4 seq = seq16(time.y, 0x2222);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = exp(-exp2(0.5) * t) * smoothstep(0.0, 0.01, q);
  //   vec2 wave = spray(t, 5400.0, 1.0, 2.0, 0.004, 64);
  //   dest += 0.08 * env * mix(0.2, 1.0, duck) * tanh(3.0 * wave);
  // }

  // { // crash
  //   float t = time.z;

  //   float env = exp(-1.0 * t);
  //   vec2 wave = spray(t, 8000.0, 0.7, 0.0, 0.001, 64);
  //   dest += 0.09 * env * mix(0.2, 1.0, duck) * tanh(8.0 * wave);
  // }

  { // grain long
    float t = tmod(time, 16.0 * B2T);

    float env = exp(-1.0 * t);
    vec2 wave = spray(t, 5000.0, 2.0, 20.0, 0.01, 8);
    dest += 0.1 * env * mix(0.3, 1.0, duck) * wave;
  }

  { // grain long
    float t = tmod(time - 8.0 * B2T, 16.0 * B2T);

    float env = exp(-1.0 * t);
    vec2 wave = spray(t, 2000.0, 0.5, 40.0, 0.02, 4);
    dest += 0.1 * env * mix(0.3, 1.0, duck) * wave;
  }

  { // grain perc
    float t0 = tmod(time, 10.0 * S2T);
    vec4 seq = seq16(t0, 0x5ac0);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp(-50.0 * t);
    vec2 wave = spray(t, 500.0, 0.7, 80.0, 0.005, 8);
    dest += 0.4 * env * mix(0.3, 1.0, duck) * tanh(3.0 * wave);
  }

  { // grain perc
    vec4 seq = seq16(time.y, 0x2424);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp(-50.0 * t);
    vec2 wave = spray(t, 2000.0, 1.0, 15.23, 0.01, 8);
    dest += 0.3 * env * mix(0.3, 1.0, duck) * tanh(3.0 * wave);
  }

  { // grain perc
    float t0 = tmod(time, 6.0 * S2T);
    vec4 seq = seq16(t0, 0x9200);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= mix(
      exp(-20.0 * t),
      exp(-t),
      0.002
    );

    vec2 wave = spray(t, 1600.0, 0.8, -14.0, 0.008, 8);
    dest += 0.3 * mix(0.3, 1.0, duck) * env * tanh(2.0 * wave);
  }

  return clip(1.3 * tanh(dest));
}
