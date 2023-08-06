#define BPM bpm
#define B2T (60.0 / bpm)

#define saturate(i) clamp(i,0.,1.)
#define clip(i) clamp(i,-1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define repeat(i,n) for(int i=0;i<(n);i++)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.0) / 12.0) * 440.0)

const float TRANSPOSE = 4.0;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;
const uint UINT_MAX = 0xffffffffu;

uvec3 pcg3d(uvec3 v) {
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

vec3 pcg3df(vec3 v) {
  uvec3 r = pcg3d(floatBitsToUint(v));
  return vec3(r) / float(UINT_MAX);
}

vec2 cis(float t) { return vec2(cos(t), sin(t)); }

mat2 rotate2D(float t) { return mat2(cos(t), sin(t), -sin(t), cos(t)); }

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
}

vec2 shotgun(float t, float spread, float snap) {
  vec2 sum = vec2(0);
  repeat(i, 64) {
    vec3 dice = pcg3df(vec3(i));

    float partial = exp2(spread * dice.x);
    partial = mix(partial, floor(partial + .5), snap);

    sum += vec2(sin(TAU * t * partial)) * rotate2D(TAU * dice.y);
  }
  return sum / 64.;
}

float cheapFilterSaw( float phase, float k ) {
  float wave = mod( phase, 1.0 );
  float c = smoothstep( 1.0, 0.0, wave / k );
  return ( wave + c ) * 2.0 - 1.0 - k;
}

vec2 cheapnoise(float t){
  uvec3 s=uvec3(t*256.);
  float p=fract(t*256.);

  vec3 dice;
  vec2 v=vec2(0);

  dice=vec3(pcg3d(s))/float(UINT_MAX)-vec3(.5,.5,0);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));
  dice=vec3(pcg3d(s+1u))/float(UINT_MAX)-vec3(.5,.5,1);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));
  dice=vec3(pcg3d(s+2u))/float(UINT_MAX)-vec3(.5,.5,2);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));

  return 2.*v;
}

mat3 orthBas(vec3 z){
  z=normalize(z);
  vec3 x=normalize(cross(vec3(0,1,0),z));
  vec3 y=cross(z,x);
  return mat3(x,y,z);
}

vec3 cyclicNoise(vec3 p, float lacu, float pers) {
  vec4 sum = vec4(0.0);
  mat3 basis = orthBas(vec3(2.0, -3.0, 1.0));
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

vec4 seq16( int seq, float st ) {
  st = mod( st, 16.0 );
  int sti = int( st ) % 16;
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - prevStepBehind;
  float nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + nextStepForward;

  return vec4(prevStep, st - prevStep, nextStep, nextStep - st);
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  const int N_CHORD = 6;
  const int N_PROG = 2;
  const int CHORDS[] = int[](
    -4, 0, 3, 7, 10, 12,
    0, 2, 5, 7, 10, 14
  );

  int prog = N_CHORD * ( int( time.z / ( 8.0 * B2T ) ) % 2 );

  float sidechain = 1.0;

  { // kick
    float t = time.x;
    float q = B2T - t;
    sidechain = smoothstep(0.0, 0.3, t) * linearstep(0.0, 0.01, q);

    float env = linearstep( 0.0, 0.0001, t ) * exp( -5.0 * t );

    vec2 phase = (
      TAU * 47.0 * t
      - 5.0 * exp( -30.0 * t )
      - 9.0 * exp( -80.0 * t )
      + 1.0 * exp( -7.0 * t ) * cheapnoise( 0.5 * t )
    );
    vec2 wave = ( tanh( 2.0 * sin( phase ) ) - 0.2 * sin( 3.0 * phase ) );

    dest += 0.6 * wave * env;
  }

  { // bass
    float t = time.x;
    float q = B2T - t;

    float env = linearstep( 0.0, 0.01, t ) + linearstep( 0.0, 0.01, q );


    float freq = p2f( 24.0 + TRANSPOSE + float( CHORDS[ prog ] ) );
    float phase = TAU * freq * t;
    vec2 wave = vec2( sin( phase ) + 0.3 * sin( 3.0 * phase ) + 0.1 * sin( 5.0 * phase ) );

    dest += 0.3 * sidechain * wave * env;
  }

  { // hihat
    float l = 0.25 * B2T;
    float t = mod( time.x, l );
    float q = l - t;

    float env = mix(
      exp( -6.0 * t ),
      exp( -60.0 * t ),
      0.94
    );
    dest += 0.1 * env * mix( 1.0, sidechain, 0.9 ) * tanh( 8.0 * shotgun( 4000.0 * t, 1.8, 0.0 ) );
  }

  { // clap
    float t = mod( time.y - B2T, 2.0 * B2T );

    float env = mix(
      mix(
        exp( -40.0 * t ),
        exp( -10.0 * t ),
        0.02
      ),
      exp( -200.0 * mod( t, 0.015 ) ),
      exp( -90.0 * max( 0.0, t - 0.02 ) )
    );

    vec3 pos = vec3( 4.0 * cis( TAU * 120.0 * t ), 140.0 * t );
    vec2 wave = cyclicNoise( pos, 2.0, 0.5 ).xy;
    dest += 0.1 * tanh( 12.0 * env * wave );
  }

  { // clav
    float l = 0.5 * B2T;
    float t = mod( time.x - 0.5 * B2T, B2T );
    float q = l - t;

    float env = linearstep( 0.0, 0.001, q ) * exp( -100.0 * t );

    vec2 wave = cis( -TAU * 9000.0 * t );

    dest += 0.1 * env * wave;
  }

  { // rim
    float t = mod( mod( time.y - B2T, 1.25 * B2T ), 0.5 * B2T );

    float env = exp( -300.0 * t );
    dest += 0.2 * env * tanh( 4.0 * (
      + tri( t * 400.0 - 0.5 * env )
      + tri( t * 1500.0 - 0.5 * env )
    ) ) * vec2( 1.0, -1.0 );
  }

  { // crash
    float t = time.z;

    float env = mix(
      exp( -t ),
      exp( -10.0 * t ),
      0.5
    );

    vec2 wave = shotgun( 4000.0 * t, 3.0, 0.0 );

    dest += 0.16 * mix( 0.2, 1.0, sidechain ) * tanh( 8.0 * wave ) * env;
  }

  { // choir
    vec2 sum = vec2( 0.0 );
    vec4 seq = seq16( 0x5252, time.y / ( 0.25 * B2T ) );
    float l = 8.0 * B2T;
    float t = mod( time.z, l );
    float q = l - t;

    float lfo = -cos( time.z / 16.0 / B2T * PI );

    const int N_UNISON = 8;
    repeat( i, N_UNISON * N_CHORD ) {
      float fi = float( i );
      vec3 dicei = pcg3df( vec3( fi ) );
      vec2 dicein = boxMuller( dicei.xy );

      float freq = p2f( 48.0 + TRANSPOSE + float( CHORDS[ i % N_CHORD + prog ] ) + 0.1 * dicein.x );
      float phase = freq * t + dicei.z;

      float env = linearstep( 0.0, 0.01, t ) * linearstep( 0.0, 0.05, q );
      vec3 pos = vec3( cis( TAU * phase ), 14.0 );
      vec2 wave = cyclicNoise( pos, 1.5, 0.2 + 0.1 * lfo ).xy * rotate2D( fi );

      sum += env * wave / float( N_UNISON * N_CHORD );
    }

    dest += 0.8 * mix( 1.0, sidechain, 0.2 ) * sum;
  }

  { // chord
    vec2 sum = vec2(0.0);
    vec4 seq = seq16(0x5252, time.y / (0.25 * B2T));
    float s = seq.s;
    float t = seq.t * (0.25 * B2T);
    float q = seq.q * (0.25 * B2T);

    vec3 dices = pcg3df(vec3(8.0 + s));

    int prog = N_CHORD * ( ( int( float( N_PROG ) + ( time.z - 0.25 * B2T ) / ( 8.0 * B2T ) ) ) % N_PROG );

    const int N_UNISON = 4;
    repeat( i, N_UNISON * N_CHORD ) {
      float fi = float(i);
      vec3 dicei = pcg3df(vec3(fi));
      vec2 dicein = boxMuller(dicei.xy);

      float freq = p2f( 48.0 + TRANSPOSE + float( CHORDS[ i % N_CHORD + prog ] ) + 0.02 * dicein.x );
      float phase = freq * t + dicei.z;

      float env = linearstep( 0.0, 0.01, t ) * linearstep( 0.0, 0.06, q );
      float k = 0.7 * ( 1.0 - exp( -3.0 * t ) );
      vec2 wave = vec2( cheapFilterSaw( phase, k ) ) * rotate2D( fi );

      sum += env * wave / float( N_UNISON * N_CHORD );
    }

    dest += 0.9 * mix( 1.0, sidechain, 0.5 ) * sum;
  }

  { // sine
    vec2 sum = vec2( 0.0 );

    const int NOTES[] = int[](
      0, 0, 0, 0, 10, 0, 5, 0,
      0, 7, 12, 5, 0, 0, 7, 0
    );

    repeat( i, 8 ) {
      float fi = float(i);

      vec4 seq = seq16( 0x8a72, ( time.y - 0.75 * B2T * fi ) / ( 0.25 * B2T ) );
      float s = seq.s;
      float t = seq.t * (0.25 * B2T);
      float q = seq.q * (0.25 * B2T);

      vec3 dicei = pcg3df( vec3( fi ) );

      float freq = p2f( 72.0 + TRANSPOSE + float( NOTES[ int( s ) ] ) );
      float phase = freq * t + dicei.z;
      // phase += 0.1 * exp( -2.0 * t ) * sin( TAU * 3.6 * phase ); // fm
      phase += 0.1 * cheapnoise( 0.002 * phase ).x;

      float env = linearstep( 0.0, 0.01, t ) * linearstep( 0.0, 0.01, q );
      env *= mix(
        exp( -5.0 * t ),
        exp( -20.0 * t ),
        0.5
      );

      vec2 wave = exp( -fi ) * vec2( sin( TAU * phase ) ) * rotate2D( 2.0 * fi );

      sum += env * wave;
    }

    dest += 0.2 * mix( 1.0, sidechain, 0.2 ) * sum;
  }

  { // arp
    vec2 sum = vec2( 0.0 );

    const int NOTES[] = int[](-5, 2, 3, 10);

    repeat( i, 4 ) {
      float fi = float( i );

      float l = 0.25 * B2T;
      float s = floor( time.z / l ) - 3.0 * fi;
      float t = mod( time.x, l );
      float q = l - t;

      vec3 dices = pcg3df( vec3( s ) );

      float freq = p2f( 48.0 + TRANSPOSE + float( NOTES[ int( mod( s, 4.0 ) ) ] ) );
      vec2 phase = vec2( t * freq ) + dices.xy;

      float env = linearstep( 0.0, 0.01, t ) * linearstep( 0.0, 0.01, q );

      float freqf = p2f( 84.0 + TRANSPOSE - 12.0 * cos( TAU * s / 64.0 ) );
      vec2 phasef = vec2( fract( phase ) / ( freq / freqf ) );

      vec2 wave = mix(
        2.0 * ( fract( phase ) - 0.5 ),
        cos( TAU * phasef ),
        exp( -0.5 * phasef ) * smoothstep( 1.0, 0.8, fract( phase ) )
      );

      sum += env * wave * exp( -fi );
    }

    dest += 0.07 * sum;
  }

  return tanh( dest );
}
