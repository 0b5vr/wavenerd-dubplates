#define TRANSPOSE -1.0
#define BPM bpm

#define beat *60.0/BPM
#define saturate(i) clamp(i, 0.,1.)
#define aSaturate(i) clamp((i),-1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(x) (asin(sin(TAU*x))/PI*2.)
#define n2f(n) (440.0*pow(2.0,((n)+TRANSPOSE)/12.0))

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;

float fs( float s ) {
  return fract( sin( s * 114.514 ) * 1919.810 );
}

vec2 wavetableSynth( float t, float freq, float rate, float offset ) {
  float phase = tri( fract( t * freq ) );
  vec2 wave = sampleSinc(
    sample_noise,
    sample_noise_meta,
    fract( 0.1 + rate * phase + offset )
  );
  return vec2( wave );
}

float kick( float t ) {
  if ( t < 0.0 ) { return 0.0; }

  float attack = 4.0;

  return exp( -4.0 * t ) * sin( TAU * (
    50.0 * t - attack * ( 2.0 * exp( -40.0 * t ) + 2.0 * exp( -200.0 * t ) )
  ) );
}

vec2 hihat( float t, float decay ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  float fmamp = -5.4 * exp( -decay * t );
  vec2 fm = fmamp * sin( vec2( 25855.0, 25955.0 ) * t );
  float amp = exp( -decay * t );
  vec2 wave = vec2( sin( fm + vec2( 8892.0, 8792.0 ) * t ) );
  wave = 2.0 * ( fract( 2.0 * wave + 0.5 ) - 0.5 );

  return amp * wave;
}

vec2 clap( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  t += 0.02 * sin( 10.0 * exp( -90.0 * t ) );
  float amp = 2.0 * exp( -30.0 * t );
  vec2 wave = sampleSinc(
    sample_noise,
    sample_noise_meta,
    lofi( t, 0.0003 )
  );

  return amp * wave;
}

vec2 ride( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  float fmamp = -0.7 * exp( -0.8 * t );
  vec2 fm = fmamp * sin( vec2( 57855.0, 57955.0 ) * t );
  float amp = exp( -1.0 * t );
  vec2 wave = vec2( sin( fm + vec2( 62892.0, 62992.0 ) * t ) );
  wave = 2.0 * ( fract( 3.0 * wave + 0.5 ) - 0.5 );

  return amp * wave;
}

vec2 crash( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  t -= 1. * exp( -40.0 * t );
  float fmamp = -8.8 * exp( -2.0 * t );
  vec2 fm = fmamp * sin( vec2( 38855.0, 38955.0 ) * t );
  float amp = exp( -1.4 * t );
  vec2 wave = vec2( sin( fm + vec2( 6792.0, 6792.0 ) * t ) );
  wave = 2.0 * ( fract( 2.3 * wave + 0.5 ) - 0.5 );

  return amp * wave;
}

vec2 snare909( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  
  vec2 wave = sampleSinc(
    sample_noise,
    sample_noise_meta,
    t
  );

  vec2 theta = t * vec2( 1500.0, 1510.0 );
  theta -= exp( -t * 80.0 ) * 30.0;
  theta -= exp( -t * 300.0 ) * 30.0;

  wave += smoothstep( -1.0, 1.0, sin( theta ) );
  
  return aSaturate( 2.0 * exp( -t * 10.0 ) * wave );
}

vec2 bass( float t, float freq ) {
  float fmamp2 = -12.4 * exp( -200.0 * t );
  vec2 fm2 = fmamp2 * vec2( sin( 2.0 * TAU * t * freq ) );
  float fmamp = -1.4 * exp( -10.0 * t );
  vec2 fm = fmamp * vec2( sin( fm2 + 2.0 * TAU * t * freq ) );
  float amp = exp( -3.0 * t );
  vec2 wave = amp * vec2( sin( fm + TAU * t * freq ) );
  return vec2( wave );
}

vec2 saw( vec2 t, float freq, vec2 offset ) {
  vec2 dest = vec2( 0.0 );
  for ( int i = 1; i <= 32; i ++ ) {
    float fi = float( i );
    dest += sin( TAU * fi * freq * t + offset ) / fi;
  }
  return dest;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );
  
  float knob0 = paramFetch( param_knob0 );
  float knob1 = paramFetch( param_knob1 );

  float tKick = time.x;
  float sidechain = smoothstep( 0.0, 0.7 beat, tKick );
  
  if ( time.z < 61.0 beat ) {
    float aKick = kick( tKick );
    dest += 0.5 * aKick;
  }
  
  { // hihat
    float t = mod( time.y, 0.25 beat );
    float decay = mix( 40.0, 100.0, fs( floor( time.z / ( 0.25 beat ) ) ) );
    float amp = 0.16 * mix( 0.3, 1.0, sidechain );
    dest += amp * hihat( t, decay );
  }
  
  { // clap
    float t = mod( time.y - 1.0 beat, 2.0 beat );
    dest += 0.3 * clap( t );
  }
  
  { // ride
    float t = time.x;
    float amp = 0.08 * mix( 0.3, 1.0, sidechain );
    dest += amp * ride( t );
  }
  
  { // crash
    float t = time.z;
    
    for ( int i = 0; i < 3; i ++ ) {
      float fi = float( i );
      float tt = t - fi beat;
      float amp = 0.37 * exp( -fi ) * mix( 0.4, 1.0, sidechain );
      dest += amp * crash( tt );
    }
  }
  
  if ( time.z > 60.0 beat ) { // snare909
    float roll = ( 0.25 - 0.125 * step( 62.0 beat, time.z ) ) beat;
    float t = mod( time.y, roll );
    float amp = mix( 0.07, 0.11, linearstep( 60.0 beat, 64.0 beat, time.z ) );
    dest += amp * snare909( t );
  }
  
  { // bass
    float notes[4] = float[]( -4.0, -4.0, 0.0, 0.0 );
    
    float t = mod( time.x - 0.5 beat, 1.0 beat );

    int prog = int( mod( time.z / ( 8.0 beat ), 4.0 ) );
    float amp = 0.5 * mix( 0.0, 1.0, sidechain );

    float freq = n2f( notes[ prog ] ) * 0.125;
    dest += amp * bass( t, freq );
  }
  
  { // pad
    float chord[20] = float[](
      -4.0, 3.0, 12.0, 17.0, 26.0,
      -4.0, 3.0, 12.0, 17.0, 26.0,
      0.0, 7.0, 14.0, 17.0, 22.0,
      0.0, 7.0, 14.0, 17.0, 22.0
    );

    float t = mod( time.z, 8.0 beat );

    float rate = 0.0004;
    rate += 0.0006 * mix(
      fs( floor( time.z / ( 0.5 beat ) ) ),
      fs( floor( time.z / ( 0.5 beat ) + 1.0 ) ),
      smoothstep( 0.0, 0.01 beat, fract( time.z / ( 0.5 beat ) ) )
    );
    rate *= ( 1.0 - exp( -0.5 * t ) );

    int prog = int( mod( time.z / ( 8.0 beat ), 4.0 ) );
    float amp = 0.03 * mix( 0.1, 1.0, sidechain );

    for ( int i = 0; i < 15; i ++ ) {
      float freq = n2f( chord[ ( i % 5 ) + 5 * prog ] ) * 0.25;
      freq *= 1.0 + 0.008 * ( 0.5 - fs( float( i ) ) );
      dest += amp * wavetableSynth( t, freq, rate, 0.03 * float( i ) );
    }
  }
  
  { // arp
    float notes[4] = float[]( 15.0, 14.0, 10.0, 5.0 );

    float t = mod( time.x, 0.25 beat );

    float rate = knob0 * 0.0002;
    rate *= exp( -mix( 40.0, 0.0, knob1 ) * t );

    int prog = int( mod( time.z / ( 0.25 beat ), 4.0 ) );
    float amp = 0.05 * mix( 0.2, 1.0, sidechain );

    for ( int i = 0; i < 6; i ++ ) {
      float freq = n2f( notes[ prog ] - 12.0 * float( i % 2 ) );
      freq *= 1.0 + 0.018 * ( 0.5 - fs( float( i ) ) );
      dest += amp * wavetableSynth( t, freq, rate, 0.03 * float( i ) );
    }
  }
  
  { // melody
  	float t = time.z;
    
    float notes[64] = float[](
      15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
      15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
      17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0,
      22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 15.0, 14.0,
      14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0,
      14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0,
      14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0,
      14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0
    );

    for ( int i = 0; i < 3; i ++ ) {
      float fi = float( i );
      float tt = t - fi beat;
      int prog = int( mod( tt / ( 0.5 beat ), 64.0 ) );

      float amp = 0.18 * exp( -fi ) * mix( 0.2, 1.0, sidechain );
      vec2 vib = ( 0.0002 + 0.0002 * fi ) * vec2(
        sin( 30.0 * tt ),
        sin( 30.0 * tt + 1.0 )
      );

      dest += 0.5 * amp * saw( tt + vib, n2f( notes[ prog ] ), vec2( 0.0 ) );
    }
  }

  return dest;
}
