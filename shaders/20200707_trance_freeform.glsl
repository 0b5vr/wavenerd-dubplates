#define TRANSPOSE 0.0

#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define P4 1.33483985417
#define P5 1.49830707688
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
  -4.0, 12.0, 14.0, 19.0,
  -5.0, 9.0, 14.0, 17.0,
  -4.0, 7.0, 12.0, 17.0,
  0.0, 10.0, 14.0, 19.0
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
  float phase = freq * t - 11.0 * ( exp( -25.0 * t ) + exp( -100.0 * t ) + exp( -700.0 * t ) );
  float fmA = sin( TAU * 1.0 * phase + 1.4 );
  vec2 fmB = 0.5 * exp( -20.0 * t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
  return aSaturate( 1.0 * vec2( exp( -4.0 * t ) * sin( TAU * phase + fmB ) ) );
}

vec2 snare( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  return aSaturate( (
    wavetable( 3.0 * t, 0.05, 0.0 ).xy +
    sin( t * 3800.0 + vec2( 0.0, 0.5 ) - exp( -t * 400.0 ) * 30.0 )
  ) * 4.0 * exp( -t * 13.0 ) );
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

vec2 snare909( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  return aSaturate( (
    noise( t ).xy * 2.0 +
    sin( t * 1400.0 * vec2( 1.005, 0.995 ) - exp( -t * 80.0 ) * 30.0 )
  ) * 1.0 * exp( -t * 10.0 ) );
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  // -- kick ---------------------------------------------------------------------------------------
  float kickTime = mod( time.x, 1.0 beat );
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

  // -- psysaw -------------------------------------------------------------------------------------
  {
    float t = mod( time.z, 0.25 beat );
    float beginStep = floor( time.z / ( 0.25 beat ) );
    vec2 dice = random2( 0.87 * beginStep );
    if ( t < ( 0.3 - dice.x * 0.2 ) beat ) {
      float freq = 5.0 * exp( 1.0 + sin( beginStep * 0.84 ) );
      dest += 0.09 * saw( 20.0 * exp( -2.0 * fract( 10.0 * exp( -freq * t ) ) ) );
    }
  }

  // -- short fx -----------------------------------------------------------------------------------
  {
    float t = mod( time.y - 0.25 beat, 2.0 beat );
    float env = exp( -t * 20.0 );
    float freq = 1400.0;
    freq *= 1.0 + mod( floor( t * 300.0 ), 3.0 );
    vec2 phase = t * freq + vec2( 0.4, 0.9 );
    vec2 pan = vec2( 0.5, 1.0 );
    pan = ( mod( time.y, 4.0 beat ) < 2.0 beat ) ? pan : pan.yx;
    dest += 0.13 * env * pan * pwm( phase, vec2( 0.3 ) );
  }

  // -- rimshot ------------------------------------------------------------------------------------
  {
    float t = mod( mod( time.x + 0.25 beat, 1.0 beat ), 0.75 beat );
    vec2 pan = vec2( 1.0, 0.5 );
    dest += 0.3 * rimshot( t );
  }

  // -- crash --------------------------------------------------------------------------------------
  {
    float amp = 0.14 * mix( 0.2, 1.0, sidechain ) * exp( -time.z );
    dest += amp * noise( time.z );
  }

  // -- bass ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y, 0.25 beat );
    vec2 dice = random2( 0.94 * lofi( time.y, 0.25 beat ) );

    float pattern[16] = float[](
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0
    );
    int iProg = int( time.z / ( 8.0 beat ) ) % 4;
    float freq = n2f( -48.0 + pattern[ int( time.y / ( 0.25 beat ) ) ] + chords[ 4 * iProg ] );
    float amp = 0.25 * sidechain;
    vec2 fm = 0.5 * exp( -6.0 * t ) * sin( 5.0 * TAU * freq * t + vec2( 1.0, 2.0 ) );
    fm += 0.5 * exp( -5.0 * t ) * sin( 6.0 * TAU * freq * t + vec2( 4.0, 5.0 ) );
    fm += 2.0 * exp( -15.0 * t ) * sin( 11.0 * TAU * freq * t + fm );
    dest += amp * aSaturate( 3.0 * sin( 1.0 * TAU * freq * t + fm ) );
  }

  // -- pads ---------------------------------------------------------------------------------------
  {
    vec2 wave = vec2( 0.0 );
    int iProg = int( time.z / ( 8.0 beat ) ) % 4;
  	vec2 diceTime = random2( 0.37 * floor( time.y / ( 0.25 beat ) ) );
    for ( int i = 0; i < 8; i ++ ) {
      vec2 dice = random2( 0.81 * float( i ) );
      float freq = n2f( chords[ ( i % 4 ) + 4 * iProg ] - 5.0 * float( i < 4 ) - 12.0 + 0.5 * ( dice.x - 0.5 ) );
      wave += (
        wavetable( time.z * freq, mix( 0.2E-4, 3E-4, diceTime.x ), -0.0001 * time.z + float( i ) )
      );
    }
    float amp = 0.07 * sidechain;
    dest += amp * wave;
  }

  // -- arp ----------------------------------------------------------------------------------------
  {
    vec2 wave = vec2( 0.0 );
    for ( int i = 0; i < 4; i ++ ) {
      float fi = float( i );
      float t = mod( time.x, 0.25 beat );
      float env = exp( -t * 10.0 );
      int iProg = int( mod( time.z - 0.75 beat * fi, 64.0 beat ) / ( 8.0 beat ) ) % 4;
      vec2 dice = random2( 0.81 * fi );
      int hit = ( int( mod( time.z - 0.75 beat * fi, 64.0 beat ) / ( 0.25 beat ) ) * 477 % 800 ) / 100; 
      float freq = n2f( chords[ ( hit % 4 ) + 4 * iProg ] + 12.0 * float( hit / 4 ) - 12.0 + 0.1 * ( dice.x - 0.5 ) );
      vec2 phase = t * freq + mix( vec2( 1.0, 0.0 ), vec2( 0.0, 1.0 ), dice.y );
      wave += env * saw( phase ) * exp( -0.7 * fi );
      wave += env * saw( phase * P5 ) * exp( -0.7 * fi );
    }
    dest += 0.17 * sidechain * wave;
  }

  return aSaturate( 1.2 * dest );
}
