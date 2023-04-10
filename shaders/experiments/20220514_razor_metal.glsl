#define BPM bpm

#define PI 3.14159265359
#define TAU 6.28318530718
#define LN2 0.69314718056
#define LN10 2.30258509299
#define clip(i) clamp(i,-1.,1.)

float decibellToVoltage( float decibell ) {
  return pow( 10.0, decibell / 20.0 );
}

vec2 razor( float freq, float time, float cutoff, float resonance ) {
  vec2 sum = vec2( 0.0 );
  for ( int i = 0; i < 96; i ++ ) {
    // osc - sawtooth
    float h = 1.0 + float( i );
    float freqH = h * freq;

    // dissonance - metal
    float hDis = pow( h, 1.2 );
    hDis = mix( hDis, floor( hDis + 0.5 ), 0.2 );
    float freqHDis = hDis * freq;

    // filter - lpf
    float octH = log( freqHDis ) / LN2;
    float octCut = log( cutoff ) / LN2;
    float octDiff = octH - octCut;
    float cutDB = min( 0.0, -24.0 * octDiff );
    cutDB += smoothstep( 0.5, 0.0, abs( octDiff ) ) * resonance;

    sum += sin( freqHDis * time * TAU ) / h * decibellToVoltage( cutDB );
  }
  return sum;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  dest += 0.3 * razor( 55.0, time.y, pow( 2.0, 14.0 - 5.0 * time.y ), 6.0 ) * exp( -time.y );

  return dest;
}
