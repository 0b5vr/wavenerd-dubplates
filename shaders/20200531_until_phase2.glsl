#define TRANSPOSE -2.0

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

float chord( float i ) {
  float iMod8 = mod( i, 8.0 );
  float note[8] = float[]( 0.0, 5.0, 10.0, 12.0, 7.0, 19.0, 17.0, 22.0 );
  return n2f( note[ int( iMod8 ) ] + 12.0 * floor( i / 8.0 ) - 24.0 );
}

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
  float attack = 20.0;
  float lorate = 0.04;

  return vec2( exp( -10.0 * t ) * sin( TAU * lofi(
    chord( -8.0 ) * t - attack * ( exp( -40.0 * t ) + exp( -200.0 * t ) ),
    lorate
  ) ) );
}

vec2 snare( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  return aSaturate( (
    noise( t ).xy +
    sin( t * 3600.0 * vec2( 1.005, 0.995 ) - exp( -t * 800.0 ) * 30.0 )
  ) * 4.0 * exp( -t * 23.0 ) );
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
    pwm( t * 1600.0 * vec2( 1.005, 0.995 ) - attack, vec2( 0.5 ) ) +
    pwm( t * 1080.0 * vec2( 0.995, 1.005 ) - attack, vec2( 0.5 ) )
  ) * sin( t * 40.0 * TAU );
  return wave * exp( -t * 20.0 );
}

vec2 bass( float freq, float t ) {
  vec2 hi = 16.0 + 0.1 * smoothstep( 0.0, 0.4, t ) * vec2( -1.0, 1.0 );
  vec2 fm = 0.1 * smoothstep( 0.0, 0.4, t ) * sin(
    TAU * freq * t + sin( TAU * hi * freq * t )
  );
  return vec2( tri( lofi( freq * t + fm, 0.0625 ) ) );
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
  float kickTime = mod( mod( time.y, 2.0 beat ), 0.75 beat );
  float sidechain = linearstep( 0.0, 0.6 beat, kickTime );
  if ( inRange( 0.0 beat, 60.75 beat, time.z ) ) {
    dest += 0.5 * kick( kickTime );
  }

  // -- hihat --------------------------------------------------------------------------------------
  float hihatTime, hihatOpen;

  {
    hihatTime = mod( time.z, 0.5 beat );
    vec2 dice = random2( time.z - hihatTime );
    float trrr = 2.2;
    hihatOpen = mix( 20.0, 200.0, dice.x );
    hihatTime = mod( hihatTime, 0.5 / pow( 2.0, floor( trrr * dice.y ) ) beat );

    float amp = 0.3 * exp( -hihatOpen * hihatTime );
    vec2 wave = wavetable( 80.0, hihatTime, 0.4, 0.0 );
    dest += amp * wave;
  }

  // -- snare --------------------------------------------------------------------------------------
  {
    dest += 0.2 * snare( mod( time.y - 1.0 beat, 2.0 beat ) );
  }

  // -- crash --------------------------------------------------------------------------------------
  {
    float amp = 0.2 * mix( 0.4, 1.0, sidechain ) * exp( -time.z );
    dest += amp * noise( time.z );
  }

  // -- cowbell ------------------------------------------------------------------------------------
  {
    dest += 0.1 * cowbell( mod( time.y - 0.25 beat, 2.0 beat ) );
  }

  // -- snare909 -----------------------------------------------------------------------------------
  if ( inRange( 60.0 beat, 64.0 beat, time.z ) ) {
    float t = time.y < 2.0 beat ? mod( time.x, 0.25 beat ) : mod( time.x, 0.125 beat );
    float amp = 0.14 * smoothstep( -4.0 beat, 4.0 beat, time.y );
    dest += amp * snare909( t );
  }

  // -- bass ---------------------------------------------------------------------------------------
  {
    dest += 0.4 * sidechain * bass( chord( -8.0 ), kickTime );
  }

  // -- pads ---------------------------------------------------------------------------------------
  {
    vec2 wave = vec2( 0.0 );
    for ( int i = 0; i < 8; i ++ ) {
      wave += (
        chordSaw( chord( 2.3 * float( i - 2 ) ), time.y )
      );
    }
    float env = 0.3;
    float amp = 0.07 * sidechain * (
      smoothstep( ( 0.1 + env ) beat, ( -0.2 + env ) beat, mod( time.z, 0.25 beat ) )
    );
    dest += amp * wave;
  }

  // -- arps ---------------------------------------------------------------------------------------
  {
    for ( int i = 0; i < 3; i ++ ) {
      float timeD = time.z - float( i ) * 0.75 beat;
      float t = mod( timeD, 0.25 beat );
      vec2 dice = random2( timeD - t + 2.59 );

      float buildup = 1.0;

      float amp = 0.14
        * exp( -mix( 40.0, 10.0, buildup ) * t )
        * exp( -mix( 4.0, 1.0, buildup ) * float( i ) );
      float freq = (
        chord( 1.3 * floor( mod( timeD / ( 0.25 beat ), 8.0 ) ) )
      );
      float form = (
        mix( 0.0002, exp( -8.0 * t ) * 0.0008, buildup )
      );

      dest += amp * wavetable( freq, t, form, dice.y );
    }
  }

  // -- chip ---------------------------------------------------------------------------------------
  {
    float freq = 2.0 * chord( floor( mod( time.y * BPM / 3.75, 8.0 ) ) );
    dest += vec2( 0.07 ) * sidechain * pwm( freq * time.y, 0.25 );
  }

  return aSaturate( dest );
}
