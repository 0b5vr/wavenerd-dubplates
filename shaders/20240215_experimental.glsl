#define B2T (60./bpm)
#define S2T (15./bpm)
#define T2S (bpm/15.)
#define ZERO min(0, int(bpm))

#define TRANSPOSE 4.0

#define saturate(x) clamp(x, 0., 1.)
#define clip(x) clamp(x, -1., 1.)
#define tri(p) (1.0 - 4.0 * abs(fract(p) - 0.5))
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float LN2 = log(2.0);
const float PI = acos(-1.0);
const float TAU = PI * 2.0;

uvec3 hash3u(uvec3 v){
  v=v*1145141919u+1919810u;
  v.x+=v.y*v.z;
  v.y+=v.z*v.x;
  v.z+=v.x*v.y;
  v^=v>>16u;
  v.x+=v.y*v.z;
  v.y+=v.z*v.x;
  v.z+=v.x*v.y;
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

vec3 cyclicNoise(vec3 p, float pump) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  repeat(i, 5) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum *= pump;
    p *= 2.;
  }

  return sum.xyz / sum.w;
}

vec4 quant(float x, float ks, float kt) {
  float i = floor(floor(x / ks + 1E-4) * ks / kt + 1E-4);
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

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float sidechain;
  float kicks = 0.0;
  float kickt = 0.0;

  { // kick
    vec4 seq = quant(mod(time.z, 64.0 * S2T), 2.0 * S2T, 5.2 * S2T);
    float t = seq.y;
    float q = seq.w;
    sidechain = 0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    kicks = seq.x;
    kickt = t;

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    float wave = sin(
      250.0 * t
      - 100.0 * exp(-t * 20.0)
      - 70.0 * exp(-t * 100.0)
      - 50.0 * exp(-t * 200.0)
    );
    dest += 0.6 * tanh(env * wave);
  }
  
  { // bass
    float t = mod(time.z, 64.0 * S2T);

    float pitch = 24.0 + TRANSPOSE;
    float freq = p2f(pitch);
    
    vec2 phase = vec2(TAU * t * freq);
    
    float radius = kickt;
    
    vec2 pm = vec2(0.0);
    repeat(i, 6) {
      float fi = float(i);
      float phasep = 2.0 * TAU * (t + sin(3.0 * t + fi) / 800.0) * freq;
      vec3 p = vec3(radius * cis(phasep), 3.0);
      pm += 0.3 * cyclicNoise(p, 2.0).xy * rotate2D(fi);
    }

    vec2 wave = vec2(0.0);
    wave += sin(1.5 * sin(phase + pm));
    
    dest += 0.5 * sidechain * wave;
  }

  { // hihat
    vec4 seq = quant(mod(time.z, 64.0 * S2T), S2T, 1.0 * S2T);
    float t = seq.y;
    float q = seq.w;

    float env = exp(-100.0 * t);
    vec2 wave = shotgun(6000.0 * t, 2.0, 0.0, 0.5);
    dest += 0.2 * env * mix(0.3, 1.0, sidechain) * tanh(8.0 * wave);
  }

  { // click
    vec4 seq = quant(mod(time.z, 64.0 * S2T), S2T, 1.62 * S2T);
    float t = seq.y;
    float q = seq.w;

    float env = exp(-2000.0 * t);

    vec2 wave = sin(1E4 * time.x + vec2(0.0, PI / 2.0));
    dest += 1.0 * mix(0.2, 1.0, sidechain) * tanh(2.0 * env * wave);
  }

  { // ride
    vec4 seq = quant(mod(time.z, 64.0 * S2T), 2.0 * S2T, 2.0 * S2T);
    float t = seq.y;
    float q = seq.w;

    float env = exp(-5.0 * t);

    vec2 sum = vec2(0.0);

    repeat(i, 8) {
      vec3 dice = hash3f(vec3(i));
      vec3 dice2 = hash3f(dice);

      vec2 wave = vec2(0.0);
      wave = 4.5 * env * sin(wave + exp2(12.10 + 0.1 * dice.x) * t + dice2.xy);
      wave = 2.2 * env * sin(wave + exp2(14.67 + 0.5 * dice.y) * t + dice2.yz);
      wave = 1.0 * env * sin(wave + exp2(13.89 + 1.0 * dice.z) * t + dice2.zx);

      sum += wave;
    }

    dest += 0.08 * env * sidechain * tanh(sum);
  }

  { // rim
    vec4 seq = quant(mod(time.z + 2.0 * S2T, 64.0 * S2T), 2.0 * S2T, 3.12 * S2T);
    float t = seq.y;
    float q = seq.w;

    float modenv = exp(-300.0 * t);
    float env = smoothstep(0.0, 0.001, t) * modenv * smoothstep(0.0, 0.01, q);

    vec2 wave = (
      + tri(400.0 * t - modenv + vec2(0.0, 0.25))
      + tri(1500.0 * t - modenv + vec2(0.0, 0.25))
    );
    dest += 0.2 * env * mix(0.3, 1.0, sidechain) * tanh(2.0 * wave);
  }

  { // beep
    vec4 seq = quant(mod(time.z, 64.0 * S2T), 2.0 * S2T, 3.42 * S2T);
    float t = seq.y;
    float q = seq.w;

    float env = smoothstep(0.0, 0.001, t) * exp(-80.0 * t);

    float phase = 5000.0 * t;
    vec2 wave = sin(8.0 * sin(TAU * phase) + TAU * phase + vec2(0.0, PI / 2.0));
    dest += 0.14 * env * mix(0.3, 1.0, sidechain) * wave;
  }

  { // chord
    const int N_CHORD = 8;
    int chord[N_CHORD] = int[](
      0, 2, 5, 7, 9, 10, 12, 14
    );

    vec2 sum = vec2(0.0);

    repeat(iUnison, 64) {
      float t = mod(time.z + float(iUnison) * S2T, 64.0 * S2T);
      float q = 64.0 * S2T - t;
      float env = smoothstep(0.0, B2T, t) * smoothstep(0.0, B2T, q);

      vec3 dice = hash3f(vec3(iUnison, 7, 7));
      vec2 dicen = boxMuller(dice.xy);

      float pitch = 48.0 + TRANSPOSE + float(chord[iUnison % N_CHORD]);
      pitch += 12.0 * floor(2.0 * dice.z);

      float phase = t * p2f(pitch) + dice.x;
      phase *= exp(0.01 * dicen.x);

      float radius = 0.3 + 0.1 * sin(3.0 * TAU * phase);
      float z = 0.1 * sin(0.1 * time.w);
      vec2 wave = (
        cyclicNoise(vec3(radius * cis(TAU * phase), 7.0 + z), 2.0).xy +
        cyclicNoise(vec3(radius * cis(TAU * 2.0 * phase), 9.0 + z), 2.0).xy
      );
      sum += 0.05 * env * wave * rotate2D(2.4 * float(iUnison));
    }

    dest += sum * sidechain * rotate2D(10.0 * time.z);
  }

  return clip(1.3 * tanh(dest));
}
