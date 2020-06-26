#define TRANSPOSE 2.0

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
#define n2f(n) (440.0*pow(2.0,((n)+TRANSPOSE)/12.0))
#define inRange(a,b,x) ((a)<=(x)&&(x)<(b))

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;

float chords[16] = float[](
  0.0, 7.0, 17.0, 22.0,
  -3.0, 4.0, 14.0, 19.0,
  -4.0, 3.0, 14.0, 19.0,
  -5.0, 5.0, 10.0, 17.0
);

vec2 noise( float t ) {
  return sampleSinc( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) );
}

vec2 random2( float t ) {
  return fract( sampleNearest( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) ) );
}

vec2 wavetable( float time, float speed, float offset ) {
  if ( time < 0.0 ) { return vec2( 0.0 ); }
  float p = tri( time );
  return noise( p * speed + offset );
}

vec2 kick( float t, float freq ) {
  float phase = freq * t - 6.0 * ( exp( -30.0 * t ) + exp( -70.0 * t ) + exp( -500.0 * t ) );
  float fmA = sin( TAU * 1.0 * phase + 1.4 );
  vec2 fmB = 0.5 * exp( -20.0 * t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
  return aSaturate( 1.0 * vec2( exp( -4.0 * t ) * sin( TAU * phase + fmB ) ) );
}

vec2 snare( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  return aSaturate( (
    noise( t ).xy +
    sin( t * 3600.0 * vec2( 1.005, 0.995 ) - exp( -t * 800.0 ) * 30.0 )
  ) * 4.0 * exp( -t * 13.0 ) );
}

vec2 snare909( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  return aSaturate( (
    noise( t ).xy * 2.0 +
    sin( t * 1400.0 * vec2( 1.005, 0.995 ) - exp( -t * 80.0 ) * 30.0 )
  ) * 1.0 * exp( -t * 10.0 ) );
}

vec2 cowbell( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  float attack = exp( -t * 800.0 ) * 20.0;
  vec2 wave = (
    pwm( t * 1600.0 * vec2( 1.005, 0.995 ) - attack, vec2( 0.125 ) ) +
    pwm( t * 1080.0 * vec2( 0.995, 1.005 ) - attack, vec2( 0.125 ) )
  ) * sin( t * 40.0 * TAU );
  return wave * exp( -t * 10.0 );
}

vec2 toms( float t, float freq ) {
  float phase = freq * t - 3.0 * ( exp( -10.0 * t ) + exp( -50.0 * t ) );
  float fmA = sin( TAU * 1.0 * phase + 1.4 );
  vec2 fmB = 0.5 * exp( -20.0 * t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
  return aSaturate( 1.0 * vec2( exp( -7.0 * t ) * sin( TAU * phase + fmB ) ) );
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

vec2 chordSaw( float freq, float t ) {
  vec2 p = fract( t * vec2( 0.99, 1.01 ) * freq );
  return (
    fract( 2.0 * p + 1.6 * sin( 2.0 * TAU * p ) ) - 0.5 +
    fract( 1.0 * p ) - 0.5
  );
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  // -- kick ---------------------------------------------------------------------------------------
  float kickTime = time.x;
  float sidechain = linearstep( 0.0, 0.6 beat, kickTime );
  if ( inRange( 0.0 beat, 61.0 beat, time.z ) ) {
    dest += 0.5 * kick( kickTime, 50.0 );
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t = mod( time.z, 0.25 beat );
    float begin = time.z - t;
    vec2 dice = random2( begin );
    float open = mix( 30.0, 200.0, dice.x );
    float amp = 0.4 * mix( 0.2, 1.0, sidechain ) * exp( -open * t );
    vec2 wave = wavetable( 40.0 * lofi( t, 0.00005 ), 4.0, 0.0 );
    dest += amp * wave;
  }

  // -- toms ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y - 0.25 beat, 2.0 beat );
    float amp = 0.1;
    dest += amp * toms( t, 120.0 ) * vec2( 0.5, 1.0 );
  }
  {
    float t = mod( time.y - 0.75 beat, 2.0 beat );
    float amp = 0.1;
    dest += amp * toms( t, 90.0 ) * vec2( 1.0, 0.5 );
  }

  // -- rimshot ------------------------------------------------------------------------------------
  {
    float t = mod( time.y - 0.25 beat, 1.25 beat );
    dest += 0.2 * rimshot( t );
  }

  // -- snare --------------------------------------------------------------------------------------
  {
    dest += 0.2 * mix( 0.6, 1.0, sidechain ) * snare( mod( time.y - 1.0 beat, 2.0 beat ) );
  }

  // -- snare909 -----------------------------------------------------------------------------------
  if ( inRange( 60.0 beat, 64.0 beat, time.z ) ) {
    float t = time.y < 2.0 beat ? mod( time.x, 0.25 beat ) : mod( time.x, 0.125 beat );
    float amp = 0.14 * smoothstep( -4.0 beat, 4.0 beat, time.y );
    dest += amp * snare909( t );
  }

  // -- cowbell ------------------------------------------------------------------------------------
  {
    dest += 0.15 * cowbell( mod( time.y - 1.25 beat, 2.0 beat ) );
  }

  // -- crash --------------------------------------------------------------------------------------
  {
    float amp = 0.14 * mix( 0.2, 1.0, sidechain ) * exp( -time.z );
    dest += amp * noise( time.z );
  }

  // -- bass ---------------------------------------------------------------------------------------
  {
    float amp = 0.3 * sidechain;
    int iProg = int( time.z / ( 8.0 beat ) ) % 4;
    float freq = n2f( -36.0 + chords[ 4 * iProg ] );
    dest += vec2( amp ) * tri( lofi( time.y * freq, 0.0625 ) );
  }

  // -- acid ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y, 0.25 beat );
    vec2 dice = random2( 0.94 * lofi( time.y, 0.25 beat ) );

    float pattern[16] = float[](
      0.0, 12.0, -2.0, 0.0,
      0.0, 0.0, 0.0, 5.0,
      0.0, 12.0, -2.0, 0.0,
      0.0, 0.0, 0.0, 5.0
    );
    int iProg = int( time.z / ( 8.0 beat ) ) % 4;
    float freq = n2f( -48.0 + pattern[ int( time.y / ( 0.25 beat ) ) ] + chords[ 4 * iProg ] );
    float amp = 0.15;
    vec2 fm = 0.5 * exp( -6.0 * t ) * sin( 5.0 * TAU * freq * t + vec2( 1.0, 2.0 ) );
    fm += 0.5 * exp( -5.0 * t ) * sin( 6.0 * TAU * freq * t + vec2( 4.0, 5.0 ) );
    fm += 2.0 * exp( -15.0 * t ) * sin( 11.0 * TAU * freq * t + fm );
    dest += amp * aSaturate( 3.0 * sin( 1.0 * TAU * freq * t + fm ) );
  }

  // -- pads ---------------------------------------------------------------------------------------
  {
    vec2 wave = vec2( 0.0 );
    float env = smoothstep( 1.0 beat, 0.1 beat, mod( mod( time.z - 0.25 beat, 2.0 beat ), 0.75 beat ) );
    // float env = smoothstep( 0.3 beat, 0.1 beat, mod( time.z, 0.25 beat ) );
    int iProg = int( time.z / ( 8.0 beat ) ) % 4;
    for ( int i = 0; i < 8; i ++ ) {
      vec2 dice = random2( 0.81 * float( i ) );
      float freq = n2f( chords[ ( i % 4 ) + 4 * iProg ] + 7.0 * float( i < 4 ) - 24.0 + 0.2 * ( dice.x - 0.5 ) );
      wave += (
        wavetable( time.z * freq, mix( 1.4E-4, 3E-4, env ), 0.0004 * time.z + float( i ) )
      );
    }
    float amp = 0.1 * sidechain * env;
    dest += amp * wave;
  }

  // -- arp ----------------------------------------------------------------------------------------
  {
    vec2 wave = vec2( 0.0 );
    float t = mod( time.x, 0.25 beat );
    float env = exp( -t * 10.0 );
    int iProg = int( time.z / ( 8.0 beat ) ) % 4;
    for ( int i = 0; i < 4; i ++ ) {
      vec2 dice = random2( 0.81 * float( i ) );
      int hit = ( int( time.z / ( 0.25 beat ) ) * 137 % 300 ) / 30; 
      float freq = n2f( chords[ ( hit % 4 ) + 4 * iProg ] + 7.0 * float( hit / 4 ) - 24.0 + 0.1 * ( dice.x - 0.5 ) );
      vec2 phase = t * freq + mix( vec2( 1.0, 0.0 ), vec2( 0.0, 1.0 ), dice.y );
      wave += (
        saw( phase + 0.04 * sin( 21.17 * TAU * phase ) )
      );
    }
    float amp = 0.12 * sidechain * env;
    dest += amp * wave;
  }

  return aSaturate( 1.2 * dest );
}
