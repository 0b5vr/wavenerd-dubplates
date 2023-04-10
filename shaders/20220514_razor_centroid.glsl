#define BPM bpm

#define PI 3.14159265359
#define TAU 6.28318530718
#define LN2 0.69314718056
#define LN10 2.30258509299
#define LN440 6.08677472691
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define f2p(i) (12.*(log(i)-LN440)/LN2+69.)
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

    // dissonance - centroid
    float octH = log( freqH ) / LN2;
    float octCent = 10.0;
    float octHDis = mix( octH, octCent, 0.2 );
    float freqHDis = pow( 2.0, octHDis );

    // filter - lpf
    float octCut = log( cutoff ) / LN2;
    float octDiff = octHDis - octCut;
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
