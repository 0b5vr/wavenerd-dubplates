#define TRANSPOSE 2.0

#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define beat *60.0/BPM
#define AMEN_BPM 170.0

#define saturate(i) clamp(i, 0.,1.)
#define aSaturate(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define n2f(n) (440.0*pow(2.0,((n)+TRANSPOSE)/12.0))

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;
uniform sampler2D sample_amen;
uniform vec4 sample_amen_meta;
uniform sampler2D sample_crash;
uniform vec4 sample_crash_meta;

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

float kick( float t, float freq ) {
  if ( t < 0.0 ) { return 0.0; }

  float attack = 8.0;
  float lorate = 0.12;

  return aSaturate( 6.0 * exp( -4.0 * t ) * sin( TAU * lofi(
    freq * t - attack * ( exp( -40.0 * t ) + exp( -200.0 * t ) ),
    lorate
  ) ) );
}

vec2 snare( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  return aSaturate( (
    noise( t ).xy * 2.0 +
    sin( t * 1400.0 * vec2( 1.005, 0.995 ) - exp( -t * 80.0 ) * 30.0 )
  ) * 2.0 * exp( -t * 10.0 ) );
}

vec2 rimshot( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  float attack = exp( -t * 400.0 ) * 0.6;
  vec2 wave = (
    tri( t * 450.0 * vec2( 1.005, 0.995 ) - attack ) +
    tri( t * 1800.0 * vec2( 0.995, 1.005 ) - attack )
  );
  return aSaturate( 4.0 * wave * exp( -t * 400.0 ) );
}

vec2 filterSaw( float freq, float time, float cutoff, float resonance ) {
  if ( time < 0.0 ) { return vec2( 0.0 ); }
  vec2 sum = vec2( 0.0 );
  for ( int i = 1; i <= 32; i ++ ) {
    float fi = float( i );
    float cut = smoothstep( cutoff * 1.2, cutoff * 0.8, fi * freq );
    cut += smoothstep( cutoff * 0.3, 0.0, abs( cutoff - fi * freq ) ) * resonance;
    vec2 offset = vec2( -1.0, 1.0 ) * ( 0.1 * ( fi - 1.0 ) );
    sum += sin( fi * freq * time * TAU + offset ) / fi * cut;
  }
  return sum;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  // -- kick ---------------------------------------------------------------------------------------
  float kickTime = mod( time.x, 1.0 beat );
  float sidechain = linearstep( 0.0, 0.6 beat, kickTime );
  {
    dest += 0.5 * kick( kickTime, n2f( -36.0 ) );
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t = mod( time.z, 0.25 beat );
    float begin = time.z - t;
    vec2 dice = random2( begin );
    float open = mix( 30.0, 200.0, dice.x );
    float amp = 0.3 * exp( -open * t );
    vec2 wave = wavetable( 20.0, lofi( t, 0.00005 ), 4.0, 0.0 );
    dest += amp * wave;
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t = mod( mod( time.y, 1.25 beat ), 0.75 beat );
    float amp = 0.05 * exp( -10.0 * t );
    vec2 wave = wavetable( 80.0, lofi( t, 0.00002 ), 0.01, 0.0 );
    dest += amp * wave;
  }

  // -- snare --------------------------------------------------------------------------------------
  {
    float t = mod( time.y - 1.0 beat, 2.0 beat );
    float amp = 0.2 * mix( 0.3, 1.0, sidechain );
    dest += amp * aSaturate( 3.0 * snare( t ) );
  }

  // -- crash --------------------------------------------------------------------------------------
  float tCrash = time.z; // time.z = 16 bars
  vec2 aCrash = sampleSinc( sample_crash, sample_crash_meta, tCrash );
  dest += 0.5 * aCrash;

  // -- amen ---------------------------------------------------------------------------------------
  {
    float t = mod( time.z, 0.5 beat );
    float begin = time.z - t;
    vec2 dice = random2( begin );
    float amenTime = ( t + floor( mod( 160.0 * dice.x, 8.0 ) ) * 0.5 beat ) / AMEN_BPM * BPM;
    vec2 aAmen = aSaturate( 20.0 * sampleSinc( sample_amen, sample_amen_meta, amenTime ) );
    dest += 0.2 * aAmen;
  }

  // -- acid ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y, 0.25 beat );
    vec2 dice = random2( 0.94 * lofi( time.y, 0.25 beat ) );
    float filt = (
      500.0 + 500.0 * cos( TAU * time.z / ( 32.0 beat ) ) +
      mix( 500.0, 3500.0, dice.x ) * exp( -mix( 10.0, 20.0, dice.y ) * t )
    );

    float pattern[16] = float[](
      0.0, 7.0, 12.0, 0.0,
      7.0, 13.0, 1.0, 8.0,
      0.0, 7.0, 12.0, 0.0,
      7.0, 13.0, 1.0, 8.0
    );
    float freq = n2f( -24.0 + pattern[ int( time.y / ( 0.25 beat ) ) ] );
    float amp = 0.25 * mix( 0.3, 1.0, sidechain );
    dest += amp * aSaturate( 3.0 * filterSaw( freq, t, filt, filt / 100.0 ) );
  }

  return aSaturate( dest );
}
