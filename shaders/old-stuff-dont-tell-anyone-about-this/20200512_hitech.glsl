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

vec2 kick( float t ) {
  return aSaturate( 1.0 * vec2( exp( -10.0 * t ) * sin( TAU * (
    50.0 * t - 6.0 * ( exp( -10.0 * t ) + exp( -80.0 * t ) + exp( -400.0 * t ) )
  ) ) ) );
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
  float kickTime = time.x;
  float sidechain = linearstep( 0.0, 0.6 beat, kickTime );
  {
    dest += 0.5 * kick( kickTime );
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
    float amp = 0.1 + 0.2 * exp( -30.0 * t );
    vec2 wave = wavetable( 14.0, t, 2.0, 0.0 );
    if ( t < 0.5 beat ) {
      dest += amp * wave;
    }
  }

  // -- psysaw -------------------------------------------------------------------------------------
  {
    float t = mod( time.z, 0.25 beat );
    float begin = time.z - t;
    vec2 dice = random2( begin );
    if ( t < ( 0.25 - dice.y * 0.2 ) beat ) {
      float freq = 20.0 * sin( TAU * begin * 2.0 );
      dest += 0.15 * saw( 20.0 * exp( -2.0 * fract( 10.0 * exp( -freq * t ) ) ) );
    }
  }

  // -- bass ---------------------------------------------------------------------------------------
  {
    // float t = mod( aTime - 0.5 beat, 1.0 beat );
    float t = mod( time.x, 0.25 beat );
    float decay = exp( -20.0 * t );
    float cutoff = mix( 100.0, 500.0, decay );
    float noteI = 0.0;
    float freq = n2f( -38.0 );
    vec2 wave = filterSaw( freq, t + 0.004 * sin( TAU * 2.0 * freq * t ), cutoff, 1.0 );
    dest += 0.4 * sidechain * decay * wave;
  }

  return aSaturate( dest );
}
