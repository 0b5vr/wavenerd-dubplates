#define TRANSPOSE -2.0
#define BPM bpm

#define beat *(60./bpm)
#define PI 3.14159265359
#define TAU 6.28318530718
#define lofi(i,j) ( floor( (i)/(j) ) * (j) )
#define n2f(n) (440.0*pow(2.0,((n)+TRANSPOSE)/12.0))
#define tri(x) (asin(sin(TAU*x))/PI*2.)

uniform vec4 param_knob0;
uniform vec4 param_knob1;

float knob;

float fs( float s ) {
  return fract( sin( s * 114.514 ) * 1919.810 );
}

vec2 kick( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  float fmamp = exp( -20.0 * t );
  
  float fmphase = 300.0 * t;
  
  float fm = fmamp * sin( TAU * fmphase );

  float amp = exp( -4.0 * t );

  float phase = 50.0 * t;
  phase -= 5.0 * exp( -40.0 * t );
  phase -= 5.0 * exp( -200.0 * t );
  
  vec2 wave = amp * vec2( sin( TAU * phase + fm ) );

  return wave;
}

vec2 hihat( float t, float decay ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  
  float amp = exp( -decay * t );
  
  float fm = sin( 189.0 * TAU * t );
  vec2 wave = 3.0 * sin( vec2( 22229.0, 22119.0 ) * TAU * t + fm );
  wave -= lofi( wave + 1.0, 2.0 );
  
  return amp * wave;
}

vec2 ride( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  
  float amp = exp( -10.0 * t );
  
  float fm = sin( 589.0 * TAU * t );
  vec2 wave = 8.8 * sin( vec2( 17229.0, 17119.0 ) * TAU * t + fm );
  wave -= lofi( wave + 1.0, 2.0 );
  
  return amp * wave;
}

vec2 snare( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  
  float amp = exp( -50.0 * t );
  
  float fm = exp( -2.0 * t ) * sin( 59.0 * TAU * t );
  vec2 wave = sin( vec2( 229.0, 219.0 ) * TAU * t + fm );
  wave = mix( wave, tri( 5.0 * wave ), 0.2 );
  
  return amp * wave;
}

vec2 snare2( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  
  float amp = exp( -30.0 * t );
  
  float fm = exp( -2.0 * t ) * sin( 39.0 * TAU * t );
  vec2 wave = sin( vec2( 329.0, 319.0 ) * TAU * t + fm );
  wave = mix( wave, tri( 12.0 * wave ), 0.2 );
  
  return amp * wave;
}

vec2 perc( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  if ( 1.5 beat < t ) { return vec2( 0.0 ); }
  
  float amp = t * t;
  
  float fm = sin( 59.0 * TAU * t );
  vec2 wave = 5.0 * sin( vec2( 279.0, 275.0 ) * TAU * t + fm );
  wave -= lofi( wave + 1.0, 2.0 );
  
  return amp * wave;
}

vec2 clap( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  
  float amp = exp( -10.0 * t );
  
  float fm = 2.0 * sin( 149.0 * TAU * t * t );
  vec2 wave = sin( vec2( 1829.0, 1834.0 ) * TAU * t + fm );
  wave = mix( lofi( wave + 0.75, 1.5 ), tri( 12.0 * wave ), 0.5 );
  
  return amp * wave;
}

vec2 bass( float t, float note, float sync ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  float knob0 = paramFetch( param_knob0 );
  
  float freq = n2f( note );
  vec2 phase = mod( fract( vec2( 0.996, 1.004 ) * freq * t ), sync ) * TAU;
  vec2 fm = exp( -20.0 * t ) * sin( 5.0 * phase );
  vec2 fm2 = 2.0 * exp( -10.0 * t ) * sin( 6.0 * phase + fm );
  vec2 wave = sin( 1.0 * phase + fm2 );
  wave = mix( wave, lofi( wave + 0.45, 0.9 ), knob0 );
  
  return exp( -3.0 * t ) * wave;
}

vec2 arp( float t, float note ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  float knob1 = paramFetch( param_knob1 );
  
  float freq = n2f( note );
  vec2 phase = fract( vec2( 0.996, 1.004 ) * freq * t ) * TAU;
  vec2 fm = exp( -10.0 * t ) * sin( ( 6.0 + 5.0 * knob1 ) * phase );
  vec2 wave = sin( 1.0 * phase + fm );
  
  return exp( -3.0 * t ) * wave;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );
  float knob0 = paramFetch( param_knob0 );
  
  float tKick = mod( time.y, 1.0 beat ); // time.x = a beat
  float sidechain = smoothstep( 0.1 beat, 0.6 beat, tKick );
  if ( time.z < 61.0 beat ) {
    vec2 aKick = kick( tKick );
    dest += 0.3 * aKick;
  }

  float tHihat = mod( mod( time.y, 0.5 beat ), 0.28 beat );
  float tbHihat = time.y - tHihat;
  vec2 aHihat = hihat( tHihat, 20.0 + 40.0 * fs( tbHihat ) );
  dest += mix( 0.3, 1.0, sidechain ) * 0.14 * aHihat;

  float tRide = mod( time.x, 0.5 beat );
  dest += mix( 0.1, 1.0, sidechain ) * 0.14 * ride( tRide );

  float tPerc = mod( time.y - 0.28 beat, 2.0 beat );
  vec2 aPerc = perc( tPerc );
  dest += mix( 0.3, 1.0, sidechain ) * 0.13 * aPerc;

  float tSnare = mod( time.y - 0.78 beat, 2.0 beat );
  dest += mix( 0.3, 1.0, sidechain ) * 0.3 * snare( tSnare );

  if ( time.z > 60.0 beat ) {
    float tSnare2 = time.y < 2.0 beat
      ? mod( mod( time.y, 0.5 beat ), 0.28 beat )
      : mod( time.y, 0.125 beat );
    dest += mix( 0.3, 1.0, sidechain ) * 0.3 * snare2( tSnare2 );
  }

  float tClap = mod( time.y - 1.0 beat, 2.0 beat );
  vec2 aClap = clap( tClap );
  dest += mix( 0.3, 1.0, sidechain ) * 0.2 * aClap;

  float tBass = mod( mod( time.y, 0.5 beat ), 0.28 beat );
  float tbBass = time.y - tBass;
  int stepBass = int( ( time.y * 4.0 ) / ( 1.0 beat ) );
  float ptnBass[16] = float[](
    -12.0, 10.0, 0.0, 12.0,
    -12.0, 0.0, 0.0, 0.0,
    -12.0, 10.0, 0.0, 12.0,
    -12.0, 0.0, 0.0, 0.0
  );
  float nBass = ptnBass[ stepBass ] - 36.0;
  dest += sidechain * 0.18 * bass( tBass, nBass, 1.0 - 0.6 * knob0 * fs( tbBass ) );

  for ( int i=0; i<3; i++ ){
    float t = time.z - 0.5 beat * float( i );
    float tArp = mod( mod( t, 0.5 beat ), 0.28 beat );
    float tbArp = t - tArp;
    float optionArp[5] = float[](
    0.0, 5.0, 7.0, 10.0, 3.0
    );
    float nArp = optionArp[ int( fs( tbArp ) * 5.0 ) ];
    nArp += 12.0 * floor( fs( tbArp + 2.0 ) * 3.0 );
    float pan = 0.3 * ( fs( tbArp + 11.0 ) - 0.5 );
    dest += sidechain * 0.09 * exp( -float( i ) ) * vec2( 0.5 + pan, 0.5 - pan ) * arp( tArp, nArp );
  }

  return dest;
}
