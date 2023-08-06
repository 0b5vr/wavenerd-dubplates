#define BPM bpm
#define B2T (60.0 / bpm)

#define saturate(i) clamp(i,0.,1.)
#define clip(i) clamp(i,-1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define repeat(i,n) for(int i=0;i<(n);i++)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.0) / 12.0) * 440.0)

const float TRANSPOSE = 0.0;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;
const uint UINT_MAX = 0xffffffffu;

const float AMEN_BPS = 170.0 / 60.0;

uniform sampler2D sample_amen;
uniform vec4 sample_amen_meta;

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

vec2 filterWave( float[128] data, float freq, vec2 phase, float cutoff, float reso ) {
  vec2 sum = vec2(0);

  for(int i=0;i<64;i++){
    float fi=float(i);
    float freqp = freq * ( 1.0 + fi );
    float omega = freqp / cutoff;
    float omegaSq=omega*omega;

    float partgain=data[i*2];
    float partarg=data[i*2+1];

    float a=4.0*reso+omegaSq*omegaSq-6.0*omegaSq+1.0;
    float b=4.0*omega*(omegaSq-1.0);
    float filtgain=1.0/sqrt(a*a+b*b);
    float filtarg=atan(a,b);

    sum += sin( ( 1.0 + fi ) * TAU * phase - TAU * partarg - filtarg ) * partgain * filtgain;
  }

  return sum;
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

  const int N_CHORD = 9;
  const int N_PROG = 4;
  const int CHORDS[] = int[](
    -12, -5, 0, 3, 7, 10, 14, 15, 17,
    -18, -11, -6, -3, 1, 4, 8, 9, 11,
    -16, -9, -4, -1, 6, 8, 10, 11, 13,
    -14, -7, -4, 0, 3, 8, 10, 12, 13
  );

  float[128] waveSaw;
  for ( int i = 0; i < 64; i ++ ) {
    waveSaw[ 2 * i ] = 0.66 / float( i + 1 );
    waveSaw[ 2 * i + 1 ] = 0.0;
  }

  float[128] waveChoir;
  for ( int i = 0; i < 64; i ++ ) {
    vec3 dice = pcg3df( vec3( i + 9 ) );
    waveChoir[ 2 * i ] = 0.5 * pow( dice.x, 3.0 ) * pow( smoothstep( 64.0, 10.0, float( i ) ), 20.0 );
    waveChoir[ 2 * i + 1 ] = dice.y;
  }

  int prog = N_CHORD * ( int( time.z / ( 8.0 * B2T ) ) % N_PROG );

  { // kick
    vec4 seq = seq16( 0x91d2, time.z / ( 0.5 * B2T ) );
    float s = seq.s;
    float t = seq.t * ( 0.5 * B2T );
    float q = seq.q * ( 0.5 * B2T );

    float env = linearstep( 0.0, 0.0001, t ) * linearstep( 0.0, 0.01, q ) * exp( -1.0 * t );

    float tr = t * exp2( 0.0 - 0.1 * s + 1.0 * step( 8.0, s ) );
    vec2 phase = vec2(
      TAU * 50.0 * tr
      - 40.0 * exp( -1.0 * tr )
      - 10.0 * exp( -10.0 * tr )
      - 3.0 * exp( -110.0 * tr )
    );
    vec2 wave = ( tanh( 2.0 * sin( phase ) ) - 0.2 * sin( 3.0 * phase ) );

    dest += 0.5 * wave * env;
  }

  { // hihat
    vec4 seq = seq16(0xffff, time.y / (0.25 * B2T));
    float s = seq.s;
    float t = seq.t * (0.25 * B2T);
    float q = seq.q * (0.25 * B2T);

    float env = exp(-60.0 * t);
    dest += 0.1 * env * tanh(8.0 * shotgun(4000.0 * t, 1.8, 0.0));
  }

  { // amen
    float s = floor( time.z / ( 0.5 * B2T ) );
    vec3 dices = pcg3df( vec3( s ) );

    float l = 0.5 * B2T;
    float t = mod( time.y, 0.5 * B2T );
    float q = l - t;

    float env = linearstep( 0.0, 0.001, t ) * linearstep( 0.0, 0.01, q );

    float slice = 0.5 * floor( 8.0 * dices.x );
    float ament = ( t / B2T + slice ) / AMEN_BPS;
    vec2 wave = clip( 4.0 * sampleSinc( sample_amen, sample_amen_meta, ament ) );
    wave -= clip( 4.0 * sampleSinc( sample_amen, sample_amen_meta, ament + 0.004 ) );

    dest += 0.2 * env * tanh( 2.0 * wave );
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
      vec2 phase = vec2( freq * t );

      float cutoff = exp2( 10.0 + 4.0 * exp( -1.0 * t ) );

      float env = linearstep( 0.0, 0.03, t ) * linearstep( 0.0, 0.05, q );
      vec2 wave = filterWave( waveChoir, freq, phase, cutoff, 0.4 );
      wave += 0.2 * sin( TAU * phase );

      sum += env * wave / float( N_UNISON * N_CHORD ) * rotate2D( 0.5 * PI + dicein.y );
    }

    dest += 3.0 * sum;
  }

  return tanh( dest );
}
