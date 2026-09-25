#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))

#define saturate(i) clamp(i, 0.0, 1.0)
#define clip(i) clamp(i, -1.0, 1.0)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define u2b(u) ((u) * 2.0 - 1.0)
#define b2u(b) ((b) * 0.5 + 0.5)
#define repeat(i, n) for(int i = ZERO; i < n; i ++)

uniform vec4 param_knob3; // kick cut

#define MACRO3 paramFetch(param_knob3)

const float TRANSPOSE = 3.0;
const float SWING = 0.58;

const float PI = acos(-1.0);
const float TAU = 2.0 * PI;

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

  float duck = 1.0 - smoothstep(0.0, 0.001, time.x) * smoothstep(0.8 * B2T, 0.0, time.x);

  { // kick
    float t = time.x;
    float q = B2T - t;
    duck = 1.0 - smoothstep(0.0, 0.001, t) * smoothstep(0.8 * B2T, 0.0, t);

    {
      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      env *= mix(
        exp2(-2.0 * t),
        exp2(-20.0 * t),
        0.1
      );

      env *= mix(1.0, exp2(-40.0 * t), MACRO3);

      float osc = tanh(3.0 * sin(
        310.0 * t - 20.0 * exp2(-40.0 * t)
        -20.0 * exp2(-800.0 * t)
      ));

      dest += 0.5 * env * osc;
    }
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s;
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, q);
    float decay = exp2(6.0 - fract(0.628 * st) - 2.0 * float(mod(st, 4.0) == 2.0));
    env *= exp(-decay * t);

    vec2 sum = vec2(0);
    for (int i = 0; i < 8; i++) {
      float fi = float(i);
      float tt = t + 0.002 * (fi + 5.0 * sin(TAU * time.z / 32.0 / B2T + 0.2 * fi));
      sum += tanh(8.0 * shotgun(5400.0 * tt, 1.4, 0.0, 1.0));
    }

    dest += 0.14 * env * tanh(sum / 4.0);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0808);
    float t = seq.y;
    float q = seq.w;

    float env = mix(
      exp2(-60.0 * t),
      exp2(-500.0 * mod(t, 0.012)),
      exp2(-100.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclic(vec3(4.0 * cis(1100.0 * t), 1240.0 * t), 0.5, 2.0).xy;

    dest += 0.12 * tanh(20.0 * env * wave);
  }

  { // jet
    vec4 seq = seq16(time.y, 0xffff);
    float st = seq.s;
    float t = seq.y;
    float q = seq.w;

    float decay = exp2(5.0 + 2.0 * fract(0.418 * st + 0.3));
    float env = exp2(-decay * t);

    float mul = exp2(2.0 * fract(0.429 * st));

    vec2 sum = vec2(0);
    for (int i = 0; i < 3; i++) {
      float fi = float(i);
      float tt = t + exp2(-11.0 + sin(TAU * time.z / 3.0 / B2T)) * fi;
      sum += cyclic(vec3(4.0 * cis(mul * 1080.0 * tt), mul * 1220.0 * tt), 1.0, 2.0).xy;
    }

    dest += 0.03 * tanh(10.0 * env * sum);
  }

  { // rim
    vec4 seq = seq16(time.y, 0x7d6f);
    float t = seq.y;

    float env = step(0.0, t) * exp2(-400.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * mix(0.8, 1.0, duck) * env * vec2(wave) * rotate2D(seq.x);
  }

  { // ride
    vec4 seq = seq16(time.y, 0xaaaa);
    float t = seq.t;

    float env = mix(
      exp2(-5.0 * t),
      exp2(-50.0 * t),
      0.2
    );

    vec2 sum = vec2(0);
    for (int i = 0; i < 4; i++) {
      float fi = float(i);
      sum += shotgun(1000.0 * t + fi * (4.0 + 8.0 * t), 4.4, 0.3, 0.0);
    }
    
    dest += 0.07 * mix(0.3, 1.0, duck) * env * tanh(10.0 * sum);
  }

  { // crash
    float t = time.z;

    float env = mix(
      exp2(-t),
      exp2(-16.0 * t),
      0.5
    );

    vec2 osc = tanh(8.0 * shotgun(4000.0 * t, 3.0, 0.0, 0.0));
    
    dest += 0.2 * mix(0.2, 1.0, duck) * env * osc;
  }

  { // ep
    const float CHORD[7] = float[](0.0, 2.0, 3.0, 7.0, 9.0, 10.0, 14.0);

    vec2 sum = vec2(0);

    for (int i = 0; i < 28; i++) {
      float delay = float(i / 7);

      float note = CHORD[i % 7];

      float t = mod(time.z, 64.0 * S2T);
      float st = mod(t2sSwing(t) - 1.0, 64.0);
      st = st < 61.0
        ? lofi(st, 3.0)
        : lofi(st, 2.0);
      st += 1.0;
      st = mod(st, 64.0);

      float prog = step(st, 7.0);

      st += 2.0 * delay;
      st = mod(st, 64.0);

      t = mod(t - s2tSwing(st), 64.0 * S2T);
      float q = (3.0 * S2T) - t;
      float qd = (1.4 * S2T) - t;

      vec3 dice = hash3f(vec3(i, st, 0));
      vec2 dicen = boxMuller(dice.xy);

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      env *= exp(-50.0 * max(-qd, 0.0));

      float freq = p2f(48.0 + prog + TRANSPOSE + note + 0.02 * dicen.y);
      float phase = lofi(freq * t + TAU * dice.z, 1.0 / 32.0);
      vec2 osc = vec2(
        + 0.25 * sin(TAU * phase)
        + 0.25 * sin(2.0 * TAU * phase)
        + 0.14 * sin(3.0 * TAU * phase)
        + 0.10 * sin(4.0 * TAU * phase)
      );

      float delaydecay = exp(-2.0 * delay);
      sum += env * delaydecay * osc * rotate2D(-3.0 * delay + 0.3 * dicen.x);
    }

    dest += 0.2 * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  dest = mainAudioDry(time);

  return clip(1.3 * tanh(dest));
}
