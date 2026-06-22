#define TRANSPOSE 0.0

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

const float SWING = 0.52;

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);

uniform vec4 param_knob3; // kick cut

#define MACRO3 paramFetch(param_knob3)

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

vec2 mainAudioDry(vec4 time) {
  vec2 dest = vec2(0);
  float duck = 1.0;

  { // kick
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);
    env *= mix(1.0, exp2(-40.0 * t), MACRO3);

    float wave = sin(TAU * (
      44.0 * t
      - 5.0 * exp2(-t * 40.0)
      - 2.0 * exp2(-t * 80.0)
    ));
    dest += 0.5 * tanh(2.0 * env * wave);
  }

  { // sub kick
    vec4 seq = seq16(time.y, 0x1212);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    float wave = sin(TAU * (
      42.0 * t
      - 4.0 * exp2(-t * 40.0)
    ));
    dest += 0.3 * (1.0 - MACRO3) * duck * tanh(2.0 * env * wave);
  }

  { // rumble
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.04, q);

    vec2 phase = TAU * vec2(
      48.0 * t
      + 0.1 * cheapnoise(1.0 * t)
    );
    vec2 wave = sin(phase);
    wave = mix(vec2(dot(wave, vec2(0.5))), wave, 0.3);

    dest += 0.3 * (1.0 - MACRO3) * mix(0.0, 1.0, duck) * env * wave;
  }

  { // rim
    vec4 seq = seq16(time.y, 0x2727);
    float t = seq.y;

    float env = exp2(-400.0 * t);

    float wave = tanh(4.0 * (
      + tri(t * 400.0 - 0.5 * env)
      + tri(t * 1500.0 - 0.5 * env)
    ));

    dest += 0.2 * mix(0.3, 1.0, duck) * env * vec2(wave) * rotate2D(seq.x);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 8.0);

    float vel = fract(st * 0.62 + 0.53);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(7.0 - 4.0 * vel) * t);

    vec2 sum = vec2(0);
    repeat(i, 4) {
      float tt = t;
      tt *= 1.0 + 0.05 * float(i);
      tt += 0.001 * float(i);

      vec2 wave = cyclic(vec3(cis(3000.0 * tt), exp2(7.0 + 2.0 * vel) * tt), 0.7, 2.0).xy;
      sum += wave * exp2(-0.4 * float(i));
    }

    dest += 0.12 * env * mix(0.3, 1.0, duck) * tanh(2.0 * sum);
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.01, q);
    env *= exp2(-60.0 * t);

    vec2 wave = shotgun(4700.0 * t, 1.8, 0.0, 0.0);
    wave = tanh(1.5 * wave);

    dest += 0.25 * mix(0.1, 1.0, duck) * env * wave;
  }

  { // ride
    vec4 seq = seq16(time.y, 0xaaaa);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    env *= exp(-4.0 * t);

    vec2 wave = shotgun(3200.0 * t, 2.0, 0.2, 1.0);

    dest += 0.06 * mix(0.3, 1.0, duck) * env * sin(4.0 * wave);
  }

  { // delay riff zone
    vec2 sum = vec2(0.0);
    repeat(iDelay, 6) {
      float fiDelay = float(iDelay);
      float delaydecay = exp(-0.8 * fiDelay);

      vec4 tdelay = mod(time - fiDelay * 3.0 * S2T, timeLength);

      vec2 sumd = vec2(0.0);
      
      { // fm bell
        vec4 seq = seq16(tdelay.y, 0x2000);
        float t = seq.t;
        float q = seq.q;

        float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
        env *= exp2(-20.0 * t);

        float freq = 340.0;
        float phase = freq * t;
        phase += exp2(-10.0 * t) * sin(TAU * phase * 2.41);

        vec2 wave = cis(TAU * phase);

        sumd += 0.07 * env * wave;
      }

      { // wavefold perc
        vec4 seq = seq16(tdelay.y, 0x0800);
        float t = seq.t;
        float q = seq.q;

        float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
        env *= exp2(-4.0 * t);

        float phase = 19.0 * exp2(-20.0 * t) * sin(TAU * 60.0 * t);
        vec2 wave = cis(TAU * (phase + 50.0 * t));

        sumd += 0.07 * env * wave;
      }

      { // beep
        vec4 seq = seq16(tdelay.y, 0x0101);
        float t = seq.t;
        float q = 0.8 * S2T - t;

        float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);

        float freq = 550.0;
        float phase = freq * t;

        vec2 wave = cis(TAU * phase);

        sumd += 0.08 * env * wave;
      }

      { // chord
        vec4 seq = seq16(tdelay.y, 0x0060);
        float t = seq.t;
        float q = seq.q;

        float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
        env *= exp2(-20.0 * t);

        float freq = 210.0;
        vec2 phase = vec2(freq * t);
        phase += vec2(0.0, 0.2);

        float k = mix(0.3, 0.96, exp2(-20.0 * t));
        vec2 wave = (
          + cheapfiltersaw(phase, k)
          - cheapfiltersaw(phase + 0.1, k)
          + cheapfiltersaw(1.09 * phase, k)
          - cheapfiltersaw(1.09 * phase + 0.1, k)
          + cheapfiltersaw(1.64 * phase, k)
          - cheapfiltersaw(1.64 * phase + 0.1, k)
        );

        sumd += 0.16 * env * wave;
      }

      { // quartal stacker
        vec4 seq = seq16(tdelay.y, 0x007f);
        float st = mod(seq.s + lofi(tdelay.z / S2T, 16.0), 256.0);
        float t = seq.t;
        float q = seq.q;
  
        if (floor(st / 16.0) == 7.0) {
          vec3 dice = hash3f(vec3(mod(st, 8.0), 55, 12));
    
          float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
          env *= mix(
            smoothstep(0.0, 0.01, 0.1 - t),
            exp2(-20.0 * t),
            0.1
          );
    
          float pitch = 24.0 + 5.0 * mod(st, 16.0);
          float freq = p2f(pitch);
          vec2 phase = vec2(freq * t);
    
          // stereo
          phase += dice.xy;
    
          vec2 wave = vec2(0.0);
          wave += sin(TAU * phase);
          wave += sin(TAU * 1.6 * phase);
    
          sumd += 0.12 * env * wave / 4.0;
        }
      }

      sum += delaydecay * sumd;
    }

    dest += mix(0.5, 1.0, duck) * sum;
  }

  { // fm drone
    float t = time.z;
    float q = timeLength.z - t;

    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    vec2 sum = vec2(0.0);
    repeat(i, 16) {
      float fi = float(i);
      vec3 dice = hash3f(vec3(fi, 58, 11));
      vec2 dicen = boxMuller(dice.xy);

      float freq = 74.0 * exp2(0.008 * dicen.x);
      float phase = freq * t + dice.z;

      phase += 0.13 * exp2(-1.0 * q) * sin(21.29 * TAU * phase);
      phase += 0.01 * sin(PI * exp2(-0.2 * t)) * sin(31.54 * TAU * phase);
      phase += 0.2 * sin(PI * exp2(-0.3 * t)) * sin(10.77 * TAU * phase);
      phase += 0.3 * sin(PI * exp2(-0.1 * t)) * sin(3.18 * TAU * phase);
      vec2 wave = cis(TAU * phase) * rotate2D(2.4 * fi);

      sum += wave / 8.0;
    }

    dest += 0.08 * mix(0.3, 1.0, duck) * env * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  dest *= 1.5;
  return dest;
}
