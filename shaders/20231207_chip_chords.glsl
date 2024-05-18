#define saturate(x) clamp(x, 0.0, 1.0)
#define linearstep(a, b, t) saturate(((t)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i) / (m)) * (m))
#define p2f(p) (440.0 * pow(2.0, (p - 69.0) / 12.0))
#define tri(x) (1.0 - 4.0 * abs(fract(x + 0.25) - 0.5))
#define repeat(i, n) for (int i = 0; i < n; i ++)

#define B2T (60.0 / bpm)

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
  uvec3 r = hash3u(floatBitsToUint(v));
  return vec3(r) / float(-1u);
}

vec2 cis(float x) {
  return sin(x + vec2(0.0, PI / 2.0));
}

mat2 r2d(float t) {
  vec2 c = cis(t);
  return mat2(c.x, c.y, -c.y, c.x);
}

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
}

vec4 seq16( int seq, float st ) {
  int sti = int( st );
  int rotated = ( ( seq >> ( 15 - sti ) ) | ( seq << ( sti + 1 ) ) ) & 0xffff;

  float prevStepBehind = log2( float( rotated & -rotated ) );
  float prevStep = float( sti ) - prevStepBehind;
  float nextStepForward = 16.0 - floor( log2( float( rotated ) ) );
  float nextStep = float( sti ) + nextStepForward;

  return vec4(
    prevStep,
    st - prevStep,
    nextStep,
    nextStep - st
  );
}

vec2 shotgun( float t, float spread, float snap ) {
  vec2 sum = vec2( 0.0 );

  for ( int i = 0; i < 64; i ++ ) {
    vec3 dice = hash3f( vec3( i ) );

    float partial = exp2( spread * dice.x );
    partial = mix( partial, floor( partial + 0.5 ), snap );

    sum += vec2( sin( TAU * t * partial ) ) * r2d( TAU * dice.y );
  }

  return sum / 64.0;
}

float cheapFilterSaw( float phase, float k ) {
  float wave = mod( phase, 1.0 );
  float c = smoothstep( 1.0, 0.0, wave / k );
  return ( wave + c ) * 2.0 - 1.0 - k;
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 up = abs(z.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 x = normalize(cross(up, z));
  return mat3(x, cross(z, x), z);
}

vec3 cyclicNoise(vec3 p, float lacu, float pers) {
  vec4 sum = vec4(0.0);
  mat3 basis = orthBas(vec3(-1.0, 2.0, -3.0));
  float amp = 1.0;

  repeat(i, 5) {
    p *= basis;

    p += sin(p.yzx);
    sum += amp * vec4(cross(cos(p), sin(p.zxy)), 1.0);

    amp *= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
}

int chords[] = int[](
  0, 16, 19, 24, 26, 31,
  -3, 9, 23, 24, 28, 31,
  -4, 15, 22, 24, 29, 31,
  -2, 10, 21, 22, 24, 29
);

vec2 mainAudio(vec4 time) {
  vec2 dest=vec2(0.0);

  float sidechain = 1.0;

  // kick
  {
    float t = time.x;
    float q = B2T - t;

    float env = linearstep( 0.0, 0.001, t ) * linearstep(0.0, 0.01, q);
    env *= linearstep( 0.25, 0.1, t );

    sidechain = smoothstep(0.0, 0.8, t / B2T) * smoothstep(1.0, 0.99, t / B2T);

    dest += 0.4 * tanh( 2.0 * env * (
      sin(
        300.0 * t
        - 60.0 * exp( -35.0 * t )
        - 40.0 * exp( -600.0 * t )
      )
    ) );
  }

  // snare
  {
    float t = mod(time.y - B2T, 2.0 * B2T);
    float q = 2.0 * B2T - t;

    vec3 p = vec3(cis(2400.0 * t), 2037.0 * t);
    vec2 wave = cyclicNoise(p, 2.0, 0.5).xy;
    wave += sin(1400.0 * t - 30.0 * exp(-t * 120.0));

    dest += 0.4 * tanh(4.0 * wave * exp(-20.0 * t));
  }

  // hihat
  {
    float t = mod(time.x, 0.25 * B2T);
    float q = 0.25 * B2T - t;

    float env = exp(-30.0 * t);

    vec2 wave = shotgun(4000.0 * t, 2.0, 0.0);

    dest += 0.4 * mix(0.4, 1.0, sidechain) * env * wave;
  }

  // bass
  {
    float t = mod(time.x, 0.5 * B2T);
    float q = 0.5 * B2T - t;
    int prog = int(time.z / (8.0 * B2T)) % 4 * 6;

    float env = exp(-10.0 * t);

    float pitch = 36.0 + float(chords[prog]);
    float freq = p2f(pitch);
    float phase = freq * t;

    vec2 wave = vec2(tri(lofi(phase, 1.0 / 32.0)));
    dest += 0.4 * sidechain * env * wave;
  }

  // modarp
  {
    vec2 sum = vec2(0.0);

    repeat(i, 3) {
      float t = time.z - B2T - float(i) * B2T;
      int prog = int(mod(t, 32.0 * B2T) / (8.0 * B2T)) % 4 * 6;

      t = mod(t, 8.0 * B2T);
      t = t < 6.0 * B2T
        ? mod(t, 1.5 * B2T)
        : (t - 6.0 * B2T);

      float q = 1.5 * B2T - t;
      float env = linearstep(0.0, 0.01, t) * linearstep(0.0, 0.01, q);
      env *= exp(-10.0 * t);
      float delayDecay = i == 0 ? 1.0 : 0.5 * exp(-float(i));

      int ARP_NOTES = 4;
      float arpPeriod = 0.0625 * B2T;
      float phase = 0.0;
      repeat(iArp, 2 * ARP_NOTES) {
        float pitch = 48.0 + float(chords[2 + (iArp % ARP_NOTES) + prog]);
        float freq = p2f(pitch);
        freq *= iArp < ARP_NOTES ? 1.0 : -1.0;
        phase += freq * clamp(mod(t, float(2 * ARP_NOTES) * arpPeriod) - float(iArp) * arpPeriod, 0.0, arpPeriod);
      }

      vec2 wave = step(0.5, fract(phase + vec2(0.0, 0.25))) - 0.5;
      sum += env * wave * r2d(float(i)) * delayDecay;
    }

    dest += 0.2 * sum;
  }

  // arp
  {
    float arpPeriod = 0.25 * B2T;
    vec2 sum = vec2(0.0);

    repeat(i, 3) {
      float t = time.z - float(i) * B2T;
      int prog = int(mod(t, 32.0 * B2T) / (8.0 * B2T)) % 4 * 6;
      float st = floor(mod(t, 8.0 * B2T) / arpPeriod);
      int iChord = prog + 1 + int(mod(1.2 * st, 5.0));
      float oct = floor(mod(1.7 * st, 3.0));

      t = mod(time.x, arpPeriod);

      float q = arpPeriod - t;
      float env = linearstep(0.0, 0.001, t) * linearstep(0.0, 0.01, q);
      env *= exp(-10.0 * t);
      float delayDecay = exp(-float(i));

      float pitch = 48.0 + 12.0 * oct + float(chords[iChord]);
      float freq = p2f(pitch);
      float phase = freq * t;

      float fmenv = exp(-10.0 * t);
      float fm = 2.0 * fmenv * sin(5.0 * TAU * phase);
      vec2 wave = cis(TAU * phase + fm);
      sum += env * wave * r2d(float(i)) * delayDecay;
    }

    dest += 0.05 * sum;
  }

  // chord
  {
    float t = mod(time.z, 8.0 * B2T);
    int prog = int(time.z / (8.0 * B2T)) % 4 * 6;
    float k = mix(0.5, 0.02, exp(-5.0 * mod(time.x, 0.5 * B2T)));

    float q = 8.0 * B2T - t;
    float env = linearstep(0.0, 0.01, t) * linearstep(0.0, 0.01, q);
    vec2 sum = vec2(0.0);

    repeat(i, 64) {
      vec3 dice = hash3f(vec3(i));
      vec2 dicen = boxMuller(dice.xy);

      float pitch = 36.0 + float(chords[i % 6 + prog]) + 0.1 * dicen.x;
      float freq = p2f(pitch);
      float phase = t * freq + dice.z;

      float wave = cheapFilterSaw(phase, k);
      sum += mix(0.3, 1.0, sidechain) * env * vec2(wave) * r2d(float(i)) / 32.0;
    }

    dest += 0.6 * sum;
  }

  return tanh(1.5 * dest);
}
