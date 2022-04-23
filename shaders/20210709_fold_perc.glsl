#define BPM bpm
#define BEAT (60.0/BPM)
#define TRANSPOSE -3.0

#define PI 3.14159265359
#define TAU 6.28318530718

#define saturate(i) clamp(i,0.,1.)
#define aSaturate(i) clamp(i,-1.,1.)
#define fs(i) (fract(sin((i)*114.514)*1919.810))
#define lofi(i,j) (floor((i)/(j))*(j))
#define n2f(n) (pow(2.0,((n)+TRANSPOSE)/12.0)*440.0)

float declick( float t ) {
  return 1.0 - exp( -max( 0.0, t ) * 4E3 );
}

float declickl( float t, float len ) {
  return declick( t ) * declick( len - t );
}

float kick( float t ) {
  if ( t < 0.0 ) { return 0.0; }

  return exp( -4.0 * t ) * sin( TAU * (
    50.0 * t - 3.0 * ( exp( -20.0 * t ) + exp( -160.0 * t ) )
  ) );
}

vec2 snare( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  return aSaturate(
    2.0 * exp( -50.0 * t ) * sin( TAU * ( 400.0 * t ) - 20.0 * exp( -200.0 * t ) ) +
    4.0 * exp( -20.0 * t ) * ( fract( sin( lofi( t + vec2( 0, 1 ), 8E-5 ) * 444.141 ) * 211.56 ) - 0.5 )
  );
}

vec2 hihat( float t, float decay ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  return exp( -decay * t ) * ( fract( sin( ( t + vec2( 0, 2 ) ) * 4444.141 ) * 15.56 ) - 0.5 );
}

vec2 filterSaw( float freq, float time, float cutoff, float resonance ) {
  if ( time < 0.0 ) { return vec2( 0.0 ); }
  vec2 sum = vec2( 0.0 );
  for ( int i = 1; i <= 32; i ++ ) {
    float fi = 0.5 + float( i );
    float cut = smoothstep( cutoff * 1.2, cutoff * 0.8, fi * freq );
    cut += smoothstep( cutoff * 0.3, 0.0, abs( cutoff - fi * freq ) ) * resonance;
    vec2 offset = vec2( -1.0, 1.0 ) * ( 0.1 * ( fi - 1.0 ) );
    sum += sin( fi * freq * time * TAU + offset ) / fi * cut;
  }
  return sum;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  float sidechain;
  
  {
    float t = time.x;
    float a = kick( t );
    dest += 0.5 * declick( BEAT - t ) * a;
    
    sidechain = smoothstep( 0.0, 0.6 * BEAT, t );
  }
  
  {
    float t = mod( time.x, 0.25 * BEAT );
    vec4 dice = fs( lofi( time.z, 0.25 * BEAT ) + 4.88 * vec4( 0, 1, 2, 3 ) );
    t = mod( t, pow( 0.5, floor( 4.0 * dice.x * dice.x ) ) * BEAT );
    float decay = mix( 100.0, 10.0, dice.y * dice.y );
    vec2 a = hihat( t, decay );
    dest += 0.2 * declickl( t, 0.25 * BEAT ) * mix( 0.5, 1.0, sidechain ) * a;
  }
  
  {
    float t = mod( time.y + BEAT, 2.0 * BEAT );
    vec2 a = snare( t );
    dest += 0.2 * declickl( t, 1.0 * BEAT ) * mix( 0.5, 1.0, sidechain ) * a;
  }
  
  {
    float t = lofi( mod( time.x, 0.25 * BEAT ), 1E-4 );
    vec4 dice = fs( lofi( time.z, 0.25 * BEAT ) + vec4( 0, 1, 2, 3 ) );
    float freq = mix( 200.0, 1000.0, dice.x * dice.x * dice.x * dice.x );
    float decay = mix( 50.0, 10.0, dice.y * dice.y );
    float fmamp = mix( 10.0, 100.0, dice.z * dice.z );
    float a = sin( fmamp * sin( freq * exp( -t ) ) * exp( -decay * t ) );
    dest += 0.1 * declickl( t, 0.25 * BEAT ) * mix( 0.5, 1.0, sidechain ) * a;
  }
  
  {
    float t = mod( time.y, 0.25 * BEAT );
    vec4 dice = fs( lofi( time.y, 0.25 * BEAT ) + 3.88 * vec4( 0, 1, 2, 3 ) );
    float filt = (
      200.0 +
      mix( 400.0, 800.0, dice.x ) * exp( -mix( 20.0, 40.0, dice.y ) * t )
    );

    float pattern[16] = float[](
      0.0, 0.0, 10.0, 0.0,
      0.0, 12.0, 0.0, 0.0,
      0.0, 0.0, 10.0, 0.0,
      0.0, 12.0, 0.0, 0.0
    );
    float freq = n2f( -24.0 + pattern[ int( time.y / ( 0.25 * BEAT ) ) ] );
    float amp = 0.3;
    vec2 a = filterSaw( freq, t, filt, filt / 1000.0 );
    dest += amp * declickl( t, 0.25 * BEAT ) * mix( 0.2, 1.0, sidechain ) * aSaturate( a );
  }
  
  {
    float t = mod( time.z, 8.0 * BEAT );
    t = lofi( t, 8E-5 );
    for ( int i = 0; i < 128; i ++ ) {
      vec4 dicei = fs( float( i ) + 2.88 * vec4( 0, 1, 2, 3 ) );
      const float chord[8] = float[](
        0.0, 5.0, 7.0, 10.0,
        14.0, 17.0, 19.0, 22.0
      );
      float note = chord[ i % 8 ];
      float freq = n2f( note ) * 0.5;
      freq *= floor( 1.0 + 7.0 * dicei.y );
      vec2 t2 = vec2( t ) + 0.01 * sin( t + dicei.x * TAU + vec2( 0, 1 ) );
      float fmmul = floor( 1.0 + 9.0 * dicei.z );
      float fmamp = 0.8 * mod( time.y / BEAT, 1.0 );
      vec2 a = sin( t2 * TAU * freq + fmamp * sin( t2 * TAU * freq * fmmul ) );
	  dest += 0.4 * declickl( t, 8.0 * BEAT ) * mix( 0.2, 1.0, sidechain ) * a / 128.0;
    }
  }

  return dest;
}
