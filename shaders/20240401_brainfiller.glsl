#define TRANSPOSE 7.0

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define saturate(i) clamp(i,0.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define repeat(i, n) for(int i=0; i<(n); i++)

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

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

vec3 cyclicNoise(vec3 p, float pers) {
  vec4 sum = vec4(0.0);

  for (int i = 0; i ++ < 4;) {
    p *= orthBas(vec3(-1.0, 2.0, -3.0));
    p += sin(p.yzx);
    sum = (sum + vec4(cross(sin(p.zxy), cos(p)), 1.0)) / pers;
    p *= 2.0;
  }

  return sum.xyz / sum.w;
}

vec2 shotgun( float t, float spread ) {
  vec2 sum = vec2( 0.0 );

  for ( int i = 0; i < 64; i ++ ) {
    vec3 dice = hash3f( vec3( i ) );
    sum += vec2( sin( TAU * t * exp2( spread * dice.x ) ) ) * rotate2D( TAU * dice.y );
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

vec2 ladderLPF(float freq, float cutoff, float reso) {
  float omega = freq / cutoff;
  float omegaSq = omega * omega;

  float a = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;
  float b = 4.0 * omega * (omegaSq - 1.0);

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

    float env = smoothstep(0.0, 0.001, q) * smoothstep(2.0 * B2T, 0.1 * B2T, t);

    // {
    //   env *= exp(-70.0 * t);
    // }

    {
      float wave = sin(
        270.0 * t
        - 40.0 * exp(-t * 20.0)
        - 20.0 * exp(-t * 60.0)
        - 10.0 * exp(-t * 300.0)
        - 0.4 * sin(120.0 * t)
      );
      dest += 0.6 * tanh(2.0 * env * wave);
    }
  }

  { // hihat
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.2 + 0.42);
    float env = exp(-exp2(7.0 - 3.0 * vel) * t);
    vec2 wave = shotgun(6000.0 * t, 2.0);
    dest += 0.2 * env * sidechain * tanh(8.0 * wave);
  }

  { // clap
    float t = mod(time.y + S2T, 4.0 * B2T);

    float env = mix(
      exp(-26.0 * t),
      exp(-200.0 * mod(t, 0.013)),
      exp(-80.0 * max(0.0, t - 0.02))
    );

    vec2 wave = cyclicNoise(vec3(4.0 * cis(800.0 * t), 1940.0 * t), 1.5).xy;

    dest += 0.1 * tanh(20.0 * env * wave);
  }

  { // shaker
    float t = mod(time.x, S2T);
    float st = mod(floor(time.y / S2T), 16.0);

    float vel = fract(st * 0.41 + 0.63);
    float env = smoothstep(0.0, 0.02, t) * exp(-exp2(6.0 - 3.0 * vel) * t);
    vec2 wave = cyclicNoise(vec3(cis(2800.0 * t), exp2(8.0 + 3.0 * vel) * t), 0.8).xy;
    dest += 0.15 * env * sidechain * tanh(2.0 * wave);
  }

  { // perc 1
    float t = mod(time.z - S2T, 6.0 * S2T);

    float env = mix(
      exp(-t),
      exp(-30.0 * t),
      0.95
    );
    vec2 wave = sin(7100.0 * t + vec2(0, PI / 2.0) + 10.0 * cheapnoise(t));
    dest += 0.3 * env * tanh(wave);
  }

  { // perc 2
    float t = mod(time.z - 3.0 * S2T, 6.0 * S2T);

    float env = mix(
      exp(-t),
      exp(-30.0 * t),
      0.95
    );
    vec2 wave = 2.0 * fract(1200.0 * t + sin(1000.0 * t) + vec2(0.0, 0.25)) - 1.0;
    dest += 0.3 * env * tanh(wave);
  }

  { // beep
    float t = mod(time.y - 5.0 * S2T, 16.0 * S2T);

    float env = smoothstep(0.0, 0.001, t) * mix(
      exp(-2.0 * t),
      smoothstep(0.0, 0.001, 0.07 - t),
      0.98
    );
    vec2 wave = sin(50000.0 * t + vec2(PI / 2.0, 0));
    dest += 0.2 * env * wave;
  }

  { // ride
    float t = mod(time.x, 2.0 * S2T);
    float q = 2.0 * S2T - t;

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

  { // crash
    float t = mod(time.z - 32.0 * B2T, 64.0 * B2T);

    float env = mix(exp(-t), exp(-10.0 * t), 0.7);
    vec2 wave = shotgun(3800.0 * t, 2.0);
    dest += 0.3 * env * sidechain * tanh(8.0 * wave);
  }

  { // additive riff
    vec2 sum = vec2(0.0);

    float t = mod(time.x, S2T);
    float q = S2T - t;
    float st = floor(time.z / S2T);
    float env = smoothstep(0.0, 0.01, t) * smoothstep(0.0, 0.01, q);

    float basefreq = 80.0;
    float stmod = fract(0.615 * st);

    float cutenv = smoothstep(0.0, 0.01, t) * exp(-14.0 * t);
    float cutoff = exp2(
      7.5
      + 3.0 * stmod
      + 4.0 * cutenv
    );

    repeat(i, 128) {
      float fi = float(i);

      float p = 1.0 + fi;
      p = pow(p, 1.1); // metal
      p = mix(p, 1.0, 0.1); // centroid
      float freq = basefreq * p;

      vec2 phase = vec2(t * freq);
      vec2 lpf = ladderLPF(freq, cutoff, 0.3);
      vec2 hpf = twoPoleHPF(freq, 200.0, 0.0);

      sum += sin(TAU * phase + lpf.y + hpf.y) / p * lpf.x * hpf.x * env * rotate2D(2.4 * fi);
    }

    dest += 0.2 * mix(0.2, 1.0, sidechain) * tanh(5.0 * sum);
  }

  { // oidos drone
    vec2 sum=vec2(0.0);

    repeat(i, 2500) {
      vec3 diceA = hash3f(vec3(i / 50));
      vec3 diceB = hash3f(vec3(i));

      float t = mod(time.z - diceA.x * (64.0 * B2T), 64.0 * B2T);
      float env = sin(PI * t / (64.0 * B2T));

      float tone = 8.0 + 9.0 * diceA.y + 0.06 * diceB.y;
      float freq = exp2(tone);
      vec2 phase = t * freq + fract(diceB.xy * 999.0);
      phase += 0.1 * fract(32.0 * phase); // add high freq

      sum += sin(TAU * phase) * env / 1000.0;
    }

    dest += 1.0 * mix(0.2, 1.0, sidechain) * sum;
  }

  return tanh(dest);
}
