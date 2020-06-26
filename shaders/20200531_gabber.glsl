#define BPM bpm
#define AMEN_BPM 170.0

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define beat *60.0/BPM
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define PI 3.14159265359
#define TAU 6.28318530718

uniform sampler2D sample_amen;
uniform vec4 sample_amen_meta;
uniform sampler2D sample_909oh;
uniform vec4 sample_909oh_meta;
uniform sampler2D sample_crash;
uniform vec4 sample_crash_meta;

float kick( float t ) {
  if ( t < 0.0 ) { return 0.0; }

  float attack = 4.0;
  float lorate = 0.06;

  return clip( 10.0 * exp( -4.0 * t ) * sin( TAU * lofi(
    52.0 * t - attack * ( exp( -40.0 * t ) + exp( -200.0 * t ) ),
    lorate
  ) ) );
}

vec2 rimshot( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }

  float attack = exp( -t * 1000.0 ) * 0.2;
  vec2 wave = (
    tri( t * 450.0 * vec2( 1.005, 0.995 ) - attack ) +
    tri( t * 1800.0 * vec2( 0.995, 1.005 ) - attack )
  );
  return clip( 4.0 * wave * exp( -t * 400.0 ) );
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  float tKick = time.x; // time.x = a beat
  float sidechain = smoothstep( 0.0, 1.0 beat, tKick );
  float aKick = kick( tKick );
  dest += 0.3 * aKick;

  float tHihat = mod( time.x - 0.5 beat, 1.0 beat );
  vec2 aHihat = sampleSinc( sample_909oh, sample_909oh_meta, tHihat );
  dest += 0.3 * aHihat;

  float tCrash = time.z; // time.z = 16 bars
  vec2 aCrash = sampleSinc( sample_crash, sample_crash_meta, tCrash );
  dest += 0.3 * aCrash;

  vec2 aRim = sidechain * rimshot( mod( time.x, 15.0 / BPM ) );
  dest += 0.2 * aRim;

  float amenTime = time.y / AMEN_BPM * BPM;
  vec2 aAmen = clip( 4.0 * sampleSinc( sample_amen, sample_amen_meta, amenTime ) );
  dest += 0.2 * aAmen;

  return dest;
}
