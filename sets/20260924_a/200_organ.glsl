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

uniform vec4 param_knob1; // sweep level
uniform vec4 param_knob3; // kick cut

#define MACRO1 paramFetch(param_knob1)
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

vec2 spray(float t, float freq, float spread, float seed, float interval, int count) {
  float grainLength = float(count) * interval;

  vec2 sum = vec2(0.0);
  repeat(i, count) {
    float fi = float(i);

    float off = -interval * fi;
    float tg = mod(t + off, grainLength);
    float prog = tg / grainLength;

    vec3 dice = hash3f(vec3(i, floor((t + off) / grainLength), seed));
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

vec2 mainAudioDry(vec4 time) {
  vec2 dest = vec2(0);

  vec3 wowp = vec3(timeLength.z * cis(TAU * time.z / timeLength.z), 0.0);
  float wow = 0.002 * cyclic(wowp, 0.5, 1.0).x;
  time = mod(time + wow, timeLength);

  float duck = 1.0;

  { // duck
    vec4 seq = seq16(time.y, 0x8888);
    float t = seq.t;
    float q = seq.q;
    duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);
  }

  // { // kick
  //   vec4 seq = seq16(time.y, 0x8888);
  //   float t = seq.t;
  //   float q = seq.q;
  //   duck = smoothstep(0.0, 0.8 * B2T, t) * smoothstep(0.0, 0.001, q);

  //   if (time.z < 60.0 * B2T) {
  //     float env = smoothstep(0.0, 0.001, q);
  //     env *= smoothstep(0.3, 0.1, t);

  //     env *= mix(1.0, exp2(-40.0 * t), MACRO3);

  //     float phase = (
  //       280.0 * t
  //       - 10.0 * exp2(-t * 40.0)
  //       - 10.0 * exp2(-t * 80.0)
  //       - 5.0 * exp2(-t * 100.0)
  //     );

  //     float osc = (
  //       sin(phase)
  //       + 0.2 * sin(1.7 * phase + 1.8)
  //       + 0.1 * sin(2.1 * phase + 2.1)
  //       + 0.1 * sin(2.2 * phase + 2.2)
  //     );

  //     dest += 0.6 * env * tanh(2.0 * osc);
  //   }
  // }

  // { // open hihat
  //   vec4 seq = seq16(time.y, 0xeeee);
  //   float st = seq.s;
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.001, q);
  //   float decay = mod(seq.s, 4.0) == 2.0 ? 20.0 : 60.0;
  //   env *= exp2(-decay * t);

  //   vec2 sum = vec2(0.0);
  //   repeat(i, 16) {
  //     float odd = float(i % 2);
  //     float tt = (t + 0.2) * mix(1.0, 1.003, odd);
  //     vec3 dice = hash3f(vec3(i / 2));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 wave = vec2(0.0);
  //     wave = 4.5 * exp2(-5.0 * t) * sin(wave + exp2(13.16 + 0.1 * dice.x) * tt + dice2.xy);
  //     wave = 3.2 * exp2(-1.0 * t) * sin(wave + exp2(11.98 + 0.3 * dice.y) * tt + dice2.yz);
  //     wave = 1.0 * exp2(-5.0 * t) * sin(wave + exp2(14.32 + 0.2 * dice.z) * tt + dice2.zx);

  //     sum += wave * mix(1.0, 0.5, odd);
  //   }

  //   dest += 0.13 * duck * env * tanh(sum);
  // }

  // { // shaker
  //   vec4 seq = seq16(time.y, 0xffff);
  //   float t = seq.t;
  //   float q = seq.q;

  //   float env = smoothstep(0.0, 0.001, q);
  //   float vel = fract(0.73 * mod(seq.s, 8.0) - 0.7);
  //   env *= smoothstep(0.0, 0.02, t) * exp2(-exp2(6.0 - vel) * t);

  //   float radius = 2.0 + 2.0 * exp(-20.0 * t);
  //   float phase = exp2(8.7 + 0.3 * vel) * t;
  //   vec2 wave = cyclic(vec3(radius * cis(TAU * phase), TAU * phase), 0.5, exp2(1.0 + vel)).xy;
  //   dest += 0.08 * env * mix(0.2, 1.0, duck) * tanh(8.0 * wave);
  // }

  // { // clap
  //   vec4 seq = seq16(time.y, 0x0808);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = mix(
  //     exp2(-60.0 * t),
  //     exp2(-300.0 * mod(t, 0.012)),
  //     exp2(-80.0 * max(0.0, t - 0.02))
  //   );

  //   vec2 wave = cyclic(vec3(4.0 * cis(1000.0 * t), 1440.0 * t), 0.5, 2.0).xy;

  //   dest += 0.2 * mix(0.8, 1.0, duck) * tanh(20.0 * env * wave);
  // }

  // { // snare
  //   vec4 seq = seq16(time.y, 0x0140);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = smoothstep(0.0, 0.001, q);
  //   float envbody = exp2(-60.0 * t);
  //   float envnoise = exp2(-60.0 * t);

  //   float freq = 250.0;
  //   float phase = freq * t;
  //   phase -= 0.01 * exp2(-800.0 * t) * freq;

  //   vec2 osc = vec2(0);
  //   osc += envbody * mix(
  //     cis(TAU * phase),
  //     cis(2.0 * TAU * phase),
  //     0.5
  //   );
  //   osc += envnoise * cheapnoise(64.0 * t);
  //   osc = mix(vec2(dot(osc, vec2(0.5))), osc, 0.3);

  //   dest += 0.12 * vec2(1.0, 0.7) * mix(0.8, 1.0, duck) * env * tanh(6.0 * osc);
  // }

  // { // perc
  //   vec4 seq = seq16(time.y, 0x0400);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = exp2(-60.0 * t);

  //   float phase = (
  //     400.0 * t
  //     - 4.0 * exp2(-t * 600.0)
  //   );

  //   vec2 osc = (
  //     cis(TAU * phase)
  //     + 0.5 * cis(1.6 * TAU * phase)
  //     + 0.8 * cis(2.2 * TAU * phase)
  //     + 0.7 * cis(2.3 * TAU * phase)
  //   );

  //   osc = osc * cheapnoise(12.0 * t);

  //   dest += 0.2 * vec2(0.5, 1.0) * mix(0.7, 1.0, duck) * env * tanh(2.0 * osc);
  // }

  // { // ride
  //   vec4 seq = seq16(time.y, 0x2222);
  //   float t = seq.y;
  //   float q = seq.w;

  //   float env = smoothstep(0.0, 0.001, q);
  //   env *= exp2(-3.0 * t);

  //   vec2 sum = vec2(0.0);

  //   repeat(i, 8) {
  //     vec3 dice = hash3f(vec3(i));
  //     vec3 dice2 = hash3f(dice);

  //     vec2 osc = vec2(0.0);
  //     osc = 2.9 * env * sin(osc + exp2(12.90 + 0.5 * dice.x) * t + dice2.xy);
  //     osc = 2.8 * env * sin(osc + exp2(12.97 + 0.5 * dice.y) * t + dice2.yz);
  //     osc = 1.0 * env * sin(osc + exp2(14.54 + 0.5 * dice.z) * t + dice2.zx);

  //     sum += osc;
  //   }

  //   dest += 0.05 * env * mix(0.3, 1.0, duck) * tanh(sum);
  // }

  { // crash
    float t = time.z;

    float env = mix(exp2(-t), exp2(-14.0 * t), 0.7);
    vec2 osc = shotgun(3500.0 * t, 2.5, 0.0, 0.0);
    dest += 0.3 * env * mix(0.1, 1.0, duck) * tanh(8.0 * osc);
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

    dest += 0.24 * MACRO1 * mix(0.4, 1.0, duck) * env * sum;
  }

  { // notes
    const int N_NOTES = 28;
    float NOTES_DURATION = s2tSwing(64.0);
    vec2 NOTES[N_NOTES] = vec2[](
      vec2(-1, -1),
      vec2(1, 5),
      vec2(4, 7),
      vec2(6, 1),
      vec2(8, 0),
      vec2(11, 0),
      vec2(14, 0),
      vec2(17, 0),
      vec2(20, 0),
      vec2(22, 4),
      vec2(23, -1),
      vec2(26, -1),
      vec2(29, -1),
      vec2(31, -1),
      vec2(33, 5),
      vec2(35, 7),
      vec2(38, 1),
      vec2(40, 0),
      vec2(43, 0),
      vec2(46, 0),
      vec2(49, 0),
      vec2(52, 0),
      vec2(54, 4),
      vec2(55, -1),
      vec2(57, -1),
      vec2(60, -1),
      vec2(63, -1),
      vec2(65, -1)
    );

    float l = NOTES_DURATION;
    float t = tmod(time, NOTES_DURATION);
    float pitchoff = 0.0;
    repeat(iNote, N_NOTES) {
      float t0 = s2tSwing(NOTES[iNote].x);
      float t1 = s2tSwing(NOTES[iNote + 1].x);
      if (t < t1) {
        l = t1 - t0;
        t -= t0;
        pitchoff = NOTES[iNote].y;
        break;
      }
    }
    float q = l - t;

    // { // bass
    //   float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
    //   env *= smoothstep(1.5 * B2T, 0.5 * B2T, t);
  
    //   float pitch = 24.0 + TRANSPOSE + pitchoff;
    //   float freq = p2f(pitch);
    //   float phase = freq * t;
    //   vec2 wave = vec2(sin(TAU * phase + 0.4 * exp2(-5.0 * t) * sin(4.0 * TAU * phase)));
  
    //   dest += 0.5 * duck * env * wave;
    // }
  
    { // stab
      const int N_CHORD = 4;
      const int CHORD[N_CHORD] = int[](
        0, 3, 7, 10
      );
  
      float dq = S2T - t;
  
      int N_OSC = 32;
      vec2 sum=vec2(0);
      repeat(i, N_OSC) {
        float fi = float(i);
        vec3 dice = hash3f(vec3(i, 28, 18));
  
        float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
        env *= mix(exp2(-1.0 * max(0.0, -dq)), exp2(-60.0 * max(0.0, -dq)), 0.9);
  
        float pitch = 60.0 + TRANSPOSE + pitchoff + float(CHORD[i % N_CHORD]);
        float freq = p2f(pitch);
        freq *= exp2(0.04 * (dice.x - 0.5));
        float phase = freq * t + dice.z;
  
        vec2 osc = vec2(0.0);
        osc += 0.4 * sin(0.5 * TAU * phase);
        osc += 0.3 * sin(TAU * phase + 1.0);
        osc += 0.4 * sin(1.5 * TAU * phase + 3.0);
        osc += 0.3 * sin(3.0 * TAU * phase + 1.0);
        osc += 0.05 * sin(5.0 * TAU * phase + 2.0);
        osc += 0.8 * exp2(-20.0 * t) * cis(3.0 * TAU * phase);
        osc += 0.5 * cyclic(vec3(2, 5, -7) * fract(0.5 * phase), 0.5, 2.0).xy;
  
        sum += env * tanh(1.0 * osc) * rotate2D(fi);
      }
  
      dest += 0.06 * mix(0.4, 1.0, duck) * sum;
    }
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  dest = mainAudioDry(time);
  // dest = mix(vec2(dot(dest, vec2(0.5))), dest, 0.2);

  return clip(1.3 * tanh(dest));
}
