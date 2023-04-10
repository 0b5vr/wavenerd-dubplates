#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define beat *60.0/BPM

#define saturate(i) clamp(i, 0.,1.)
#define aSaturate(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define n2f(n) (440.0*pow(2.0,(n)/12.0))

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;

vec2 noise( float t ) {
  return sampleSinc( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) );
}

vec2 random2( float t ) {
  return fract( sampleNearest( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) ) );
}

vec2 wavetable( float freq, float time, float speed, float offset ) {
  if ( time < 0.0 ) { return vec2( 0.0 ); }
  float p = tri( freq * time );
  return noise( p * speed + offset );
}

vec2 kick( float t, float freq ) {
  float phase = freq * t - 12.0 * ( exp( -30.0 * t ) + exp( -80.0 * t ) + exp( -500.0 * t ) );
  float fmA = sin( TAU * 2.0 * phase + 1.4 );
  vec2 fmB = 0.3 * smoothstep( 0.25 beat, 0.75 beat, t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
  return aSaturate( 20.0 * vec2( exp( -1.0 * t ) * saw( phase + fmB ) ) );
}

// vec2 kick( float t, float freq ) {
//   float phase = freq * t - 7.0 * ( exp( -40.0 * t ) + exp( -100.0 * t ) + exp( -500.0 * t ) );
//   float fmA = sin( TAU * 2.0 * phase + 1.4 );
//   vec2 fmB = 0.7 * smoothstep( 0.25 beat, 0.75 beat, t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
//   return aSaturate( 10.0 * vec2( exp( -1.0 * t ) * sin( TAU * phase + fmB ) ) );
// }

vec2 snare909( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  return aSaturate( (
    noise( t ).xy * 2.0 +
    sin( t * 1600.0 * vec2( 1.005, 0.995 ) - exp( -t * 200.0 ) * 30.0 )
  ) * 1.0 * exp( -t * 10.0 ) );
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  // -- kick ---------------------------------------------------------------------------------------
  float kickTime = time.x;
  float sidechain = linearstep( 0.0, 0.6 beat, kickTime );
  float ring = 0.0 * sin( 800.0 * TAU * kickTime ) / 400.0;
  {
    dest += 0.5 * kick( kickTime + ring, n2f( -32.0 ) );
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t = mod( time.z, 0.25 beat );
    float begin = time.z - t;
    vec2 dice = random2( begin );
    float open = mix( 30.0, 200.0, dice.x );
    float amp = 0.3 * exp( -open * t );
    vec2 wave = wavetable( 20.0, t, 4.0, 0.0 );
    dest += amp * wave;
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t = mod( time.x - 0.5 beat, 1.0 beat );
    float amp = 0.3 * exp( -30.0 * t );
    vec2 wave = wavetable( 100.0, t, 0.8, 0.0 );
    dest += amp * wave;
  }

  // -- clap ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y - 1.0 beat, 2.0 beat );
    float amp = 0.3;
    dest += amp * aSaturate( 2.0 * snare909( t ) );
  }

  return aSaturate( dest );
}
