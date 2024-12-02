// shoutouts to Atsuo The Pineapple Donkey!
// https://www.tiktok.com/@atsuothepineappledonkey/video/7352147776601427201

#define TRANSPOSE 4.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(i) clamp(i,0.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define repeat(i, n) for(int i=ZERO; i<(n); i++)

const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float LN2 = log(2.0);
const float P4 = pow(2.0, 5.0 / 12.0);
const float P5 = pow(2.0, 7.0 / 12.0);

uvec3 hash3u(uvec3 v) {
  v = v * 1664525u + 1013904223u;

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
  uvec3 u = hash3u(floatBitsToUint(v));
  return vec3(u) / float(-1u);
}

vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D( float x ) {
  vec2 v = cis(x);
  return mat2(v.x, v.y, -v.y, v.x);
}

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
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

vec4 seq16( int seq, float t, float tps ) {
  int sti = int( t / tps ) & 15;
  int rotated = ( ( seq >> ( 15 - sti ) ) | ( seq << ( sti + 1 ) ) ) & 0xffff;

  float prevStepBehind = log2( float( rotated & -rotated ) );
  float prevStep = float( sti ) - prevStepBehind;
  float nextStepForward = 16.0 - floor( log2( float( rotated ) ) );
  float nextStep = float( sti ) + nextStepForward;

  return vec4(
    prevStep,
    mod( t, 16.0 * tps ) - prevStep * tps,
    nextStep,
    nextStep * tps - mod( t, 16.0 * tps )
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

  return vec4(s / kt, t, (s + l) / kt, q);
}

vec4 quant(float x, float ks, float kt) {
  float i;
  return quant(x, ks, kt, i);
}

float swing(float x, float k) {
  float xm = mod(x, 2.0);
  return x + (1.0 - k) * linearstep(0.0, k, xm) * linearstep(2.0, k, xm);
}

float unswing(float x0, float x, float y, float k) {
  return (
    x0
    - 2.0 * floor((x - y) / 2.0)
    - k * linearstep(0.0, 1.0, mod(x - y, 2.0))
    - (2.0 - k) * linearstep(1.0, 2.0, mod(x - y, 2.0))
  );
}

float cheapFilterSaw( float phase, float k ) {
  float i_wave = fract( phase );
  float i_c = smoothstep( 1.0, 0.0, i_wave / k );
  return ( i_wave + i_c ) * 2.0 - 1.0 - k;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float sidechain;


  { // kick
    float t = time.x;
    float q = B2T - t;
    sidechain = 0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    {
      float wave = sin(
        250.0 * t
        - 30.0 * exp(-t * 20.0)
        - 30.0 * exp(-t * 30.0)
        - 10.0 * exp(-t * 300.0)
      );
      dest += 0.6 * tanh(1.0 * env * wave);
    }
  }

  { // bass
    int notes[] = int[](
      0, 10, 12, 0,
      0, 0, 0, 0,
      0, 10, 12, 0,
      0, 0, 0, 0
    );

    vec4 seq = seq16(0xffff, time.y, S2T);
    float s = mod(round(seq.s), 16.0);
    float t = seq.t;
    float q = seq.q;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q) * exp(-20.0 * t);

    float pitch = 24.0 + TRANSPOSE + float(notes[int(s)]);
    float freq = p2f(pitch);

    vec2 phase = vec2(t * freq);
    phase += 0.1 * exp(-20.0 * t) * vec2(1.0, -1.0) * cos(2.0 * TAU * phase);

    vec2 wave = vec2(sin(TAU * phase) - 0.2 * sin(3.0 * TAU * phase));
    wave = tanh(3.0 * sin(3.0 * wave));

    // sub bass
    wave += tanh(sin(TAU * freq * t));

    dest += 0.3 * env * sidechain * wave;
  }

  { // hihat
    vec4 seq = seq16(0x2222, time.y, S2T);
    float t = seq.t;
    float q = seq.q;

    float env = mix(
      exp(-70.0 * t),
      exp(-4.0 * t),
      0.01
    );
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 1.0);
    dest += 0.2 * env * sidechain * tanh(8.0 * wave);
  }

  { // perc
    vec2 sum = vec2(0.0);
    for (int i = 0; i < 3; i ++) {
      float fi = float(i);
      float delayDecay = exp(-1.6 * fi);

      vec4 seq = quant(time.y - 3.0 * fi * S2T, S2T, 1.81 * S2T);
      float s = mod(round(seq.s), 16.0);
      float t = seq.t;
      float q = seq.q;

      float env = exp(-140.0 * t);

      float pitch = fract(0.604 * s);

      float phase = 2.0 * exp2(2.0 * pitch) * exp(-40.0 * t);
      phase += exp(-0.7 * fi) * exp(-300.0 * t) * sin(7.0 * TAU * phase);
      vec2 wave = cis(TAU * phase);
      sum += env * delayDecay * tanh(2.0 * wave);
    }
    dest += 0.1 * sidechain * sum;
  }

  { // psysaw
    vec2 sum = vec2(0.0);
    for (int i = 0; i < 6; i ++) {
      float fi = float(i);
      float delayDecay = exp(-0.7 * fi);
      float rot = 2.4 * fi;

      float t = mod(time.z - 32.0 * B2T - 3.0 * fi * S2T, 64.0 * B2T);
      float env = smoothstep(0.0, 0.01, t) * exp(-10.0 * t);
  
      vec2 phase = 40.0 * exp(-50.0 * mod(t, 0.02)) * exp(-4.0 * t) + 0.01 * sin(time.w + vec2(0.0, PI));
      vec2 wave = 2.0 * fract(phase) - 1.0;

      sum += env * delayDecay * vec2(wave) * rotate2D(rot);
    }
    dest += 0.05 * sum;
  }

  { // crash
    float t = time.z;

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0, 0.0, 3.0);
    dest += 0.1 * env * mix(0.3, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // sine stab
    int notes[] = int[](
      0, 10, 0, 7, 14, 5
    );

    vec2 sum = vec2(0.0);
    for (int i = 0; i < 4; i ++) {
      float fi = float(i);
      float delayDecay = exp(-fi);

      float seqi1, seqi2;
      vec4 seq1 = quant(time.z - 2.0 * fi * S2T, S2T, 3.41 * S2T, seqi1);
      vec4 seq2 = quant(time.z - 2.0 * fi * S2T, S2T, 5.18 * S2T, seqi2);
      float t = min(seq1.t, seq2.t);
      float q = min(seq1.q, seq2.q);
      float seqi = mod(round(seqi1 + seqi2), 6.0);
  
      float tt = max(t - 0.17 * B2T, 0.0);
      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.01, q) * mix(
        exp(-70.0 * tt),
        exp(-4.0 * tt),
        0.01
      );
  
      float pitch = 60.0 + TRANSPOSE + float(notes[int(seqi)]);
      float freq = p2f(pitch);
      float phase = t * freq;
  
      vec2 wave = (
        + cis(TAU * phase)
        + cis(P5 * TAU * phase)
      );
      
      sum += env * delayDecay * wave * rotate2D(time.w);
    }
    dest += 0.07 * mix(0.4, 1.0, sidechain) * sum;
  }

  { // chord
    const int N_CHORD = 8;
    int CHORD[N_CHORD] = int[](
      0, 7, 9, 10, 14, 17, 21, 22
    );

    vec2 sum = vec2(0.0);

    vec4 seqg = quant(mod(time.z / S2T, 64.0), 1.0, 1.28);

    repeat(iUnison, 64) {
      float t = mod(time.y, 4.0 * B2T);
      float q = 4.0 * B2T - t;
      float intensity = mix(0.1, 1.0, smoothstep(0.0, 4.0 * B2T, t));
      float amp = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q) * intensity;

      vec3 dice = hash3f(vec3(iUnison, 7, 7));
      vec2 dicen = boxMuller(dice.xy);

      float pitch = 48.0 + TRANSPOSE + float(CHORD[iUnison % N_CHORD]);
      float freq = p2f(pitch);

      float phase = t * freq + dice.x;
      phase *= exp(0.01 * dicen.x);
      phase += 0.0001 * freq * sin(30.0 * t + dice.x);

      float pers = mix(0.1, 0.4, intensity);

      vec3 c = vec3(0.0);
      vec3 d = vec3(2.0, -3.0, -8.0);

      vec2 wave = mix(
        cyclic(fract(phase) * d, pers, 2.0),
        cyclic((fract(phase) - 1.0) * d, pers, 2.0),
        smoothstep(0.9, 1.0, fract(phase))
      ).xy;

      sum += amp * wave * rotate2D(float(iUnison));
    }

    dest += 0.01 * sum * mix(0.2, 1.0, sidechain);
  }

  return tanh(1.5 * dest);
}
