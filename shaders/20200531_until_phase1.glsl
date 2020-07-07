#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define beat *60.0/BPM
#define TRANSPOSE 0.0

#define saturate(i) clamp(i, 0.,1.)
#define aSaturate(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define n2f(n) (440.0*pow(2.0,(n)/12.0))
#define inRange(a,b,x) ((a)<=(x)&&(x)<(b))

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;

float chord( float i ) {
  float iMod8 = mod( i, 8.0 );
  float note[8] = float[]( 0.0, 5.0, 10.0, 12.0, 7.0, 19.0, 17.0, 22.0 );
  return n2f( note[ int( iMod8 ) ] + 12.0 * floor( i / 8.0 ) - 24.0 + TRANSPOSE );
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
  float attack = 3.0;

  return vec2( exp( -10.0 * t ) * sin( TAU * (
    chord( -8.0 ) * t - attack * ( exp( -40.0 * t ) + exp( -200.0 * t ) )
  ) ) );
}

vec2 clap( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  float amp = exp( -14.0 * t );
  amp *= mix(
    fract( 1.0 - 30.0 * t ),
    1.0,
    smoothstep( 0.08, 0.09, t )
  );
  vec2 wave = wavetable( 4.0, t, 0.03, 0.0 );
  return lofir( amp * wave, 0.25 );
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

vec2 bass( float freq, float t ) {
  vec2 hi = 16.0 + 0.1 * smoothstep( 0.0, 0.4, t ) * vec2( -1.0, 1.0 );
  vec2 fm = 0.1 * smoothstep( 0.0, 0.4, t ) * sin(
    TAU * freq * t + sin( TAU * hi * freq * t )
  );
  return vec2( tri( lofi( freq * t + fm, 0.0625 ) ) );
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
  float kickTime = mod( mod( time.y, 2.0 beat ), 0.75 beat );
  float sidechain = linearstep( 0.0, 0.6 beat, kickTime );
  {
    dest += 0.5 * kick( kickTime );
  }

  // -- hihat --------------------------------------------------------------------------------------
  float hihatTime, hihatOpen;

  {
    hihatTime = mod( time.z, 0.5 beat );
    vec2 dice = random2( time.z - hihatTime );
    float trrr = 1.9;
    hihatOpen = 200.0;
    hihatTime = mod( hihatTime, 0.5 / pow( 2.0, floor( trrr * dice.y ) ) beat );

    float amp = 0.3 * exp( -hihatOpen * hihatTime );
    vec2 wave = wavetable( 80.0, hihatTime, 0.4, 0.0 );
    dest += amp * wave;
  }

  // -- rimshot ----------------------------------------------------------------
  {
    vec2 t = vec2(
      mod( mod( mod( time.z, 2.25 beat ), 1.25 beat ), 0.5 beat ),
      mod( mod( mod( time.z - 0.25 beat, 2.75 beat ), 0.75 beat ), 0.5 beat )
    );
    dest.x += 0.3 * rimshot( t.x ).x;
    dest.y += 0.3 * rimshot( t.y ).y;
  }

  // -- clav -------------------------------------------------------------------
  {
    float t = mod( mod( mod( time.z - 0.75 beat, 3.75 beat ), 2.75 beat ), 0.75 beat );
    float amp = exp( -100.0 * t );
    float phase = t + tri( 700.0 * t );
    dest += 0.2 * amp * tri( phase * vec2( 0.9, 1.1 ) );
  }

  // -- clap ---------------------------------------------------------------------------------------
  {
    dest += 0.2 * clap( mod( time.y - 1.0 beat, 2.0 beat ) );
  }

  // -- crash ------------------------------------------------------------------
  {
    float amp = 0.2 * mix( 0.4, 1.0, sidechain ) * exp( -time.z );
    dest += amp * noise( time.z );
  }

  // -- acid -------------------------------------------------------------------
  {
    if ( mod( time.x, 0.25 beat ) < 0.2 beat ) {
      float t = mod( time.z, 0.25 beat );
      vec2 dice = random2( 0.94 * lofi( time.z, 0.25 beat ) + 4.12 );
      vec2 dice2 = random2( 0.94 * lofi( time.z, 0.25 beat ) + 2.25 );
      float filt = (
        100.0 +
        mix( 1000.0, 4000.0, dice2.x ) * exp( -mix( 20.0, 40.0, dice2.y ) * t )
      );
      float freq = (
        chord( -8.0 + 8.0 * pow( dice.y, 5.0 ) ) *
        ( dice.x < 0.2 ? 1.335 : 1.0 ) *
        ( dice2.y < 0.5 ? 2.0 : 1.0 )
      );
      float amp = 0.16 * smoothstep( 0.25 beat, 0.2 beat, t );
      dest += amp * aSaturate( 0.7 * filterSaw( freq, t, filt, filt / 500.0 ) );
    }
  }

  // -- pads -------------------------------------------------------------------
  {
    vec2 wave = vec2( 0.0 );
    for ( int i = 0; i < 8; i ++ ) {
      wave += (
        vec2( sin( TAU * lofi(
          chord( float( i ) ) * time.y + sin( float( i ) + 2.0 * time.y ),
          0.0625
        ) ) )
      );
    }
    float amp = 0.07 * sidechain;
    dest += amp * wave;
  }

  // -- arps ---------------------------------------------------------------------------------------
  {
    for ( int i = 0; i < 3; i ++ ) {
      float timeD = time.z - float( i ) * 0.75 beat;
      float t = mod( timeD, 0.25 beat );
      vec2 dice = random2( timeD - t + 2.59 );

      float buildup = 0.0;

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

  return aSaturate( dest );
}
