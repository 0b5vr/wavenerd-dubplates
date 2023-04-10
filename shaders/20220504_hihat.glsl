#define BPM bpm
#define beat *(60.0/BPM)

#define PI 3.14159265359
#define TAU 6.28318530718

#define saw(p) (1.-2.*fract(p))
#define tri(p) (1.-4.*abs(fract(p+0.25)-0.5))
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)

float kick( float t ) {
  float attack = 4.0;

  return exp( -4.0 * t ) * sin( TAU * (
    50.0 * t - exp( -40.0 * t )
  ) );
}

// sine pm modulated pwm
float hihat( float t ) {
  float attack = 4.0;

  float hitPm = 0.4 * sin( 9000.0 * t );
  float hit = pwm( 40000.0 * t + hitPm, 0.5 );

  return exp( -200.0 * t ) * hit;
}

// saw pm modulated tri + fold
float hihat2( float t ) {
  float attack = 4.0;

  float hitPm = saw( 7350.0 * t ) * exp( -100.0 * t );
  float hit = tri( 3000.0 * t - hitPm );
  float sinHit = tri( mix( 0.2, 2.0, exp( -240.0 * t ) ) * hit );

  return exp( -200.0 * t ) * sinHit;
}

// "fract sin"
float hihat3( float t ) {
  float attack = 4.0;

  float hit = 1.0 - 2.0 * fract( 0.5 + 60.0 * sin( 4444.0 * t ) );

  return exp( -200.0 * t ) * hit;
}

// lcg
float hihat4( float t ) {
  float attack = 4.0;

  uint lcg = ( uint( t * 48000.0 ) * 1103515245u );
  float hit = ( float( lcg ) / 2147483647.0 ) - 1.0;

  return exp( -200.0 * t ) * hit;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  {
    float t = time.x;
    dest += 0.5 * kick( t );
  }

  {
    float t = mod( time.y - 0.5 beat, 4.0 beat );
    dest += 0.5 * hihat( t );
  }

  {
    float t = mod( time.y - 1.5 beat, 4.0 beat );
    dest += 0.5 * hihat2( t );
  }

  {
    float t = mod( time.y - 2.5 beat, 4.0 beat );
    dest += 0.5 * hihat3( t );
  }

  {
    float t = mod( time.y - 3.5 beat, 4.0 beat );
    dest += 0.5 * hihat4( t );
  }

  return dest;
}
