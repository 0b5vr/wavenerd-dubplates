#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define BPS (BPM / 60.0)
#define TIME2BEAT BPS
#define BEAT2TIME (1.0 / BPS)

#define clip(i) clamp(i, -1.,1.)
#define tri(p) (1.-4.*abs(fract(p+0.25)-0.5))

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;

// == utils ========================================================================================
vec2 noise( float t ) {
  return sampleSinc( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) );
}

// == euclidean rhythms ============================================================================
bool euclideanRhythms( float pulses, float steps, float i ) {
    float t = mod( i * pulses, steps );
    return t - pulses < 0.0;
}

float euclideanRhythmsInteg( float pulses, float steps, float time ) {
    float t = mod( floor( time ) * pulses, steps );
    return floor( ( t - pulses ) / pulses ) + 1.0 + fract( time );
}

// == percussions ==================================================================================
vec2 kick( float t ) {
    float phase = 45.0 * t - 6.0 * exp( -40.0 * t ) - 3.0 * exp( -400.0 * t );
    float decay = exp( -3.0 * t );
    return vec2( decay * sin( TAU * phase ) );
}

vec2 hihat( float t ) {
    float decay = exp( -50.0 * t );
    vec2 sig = noise( 0.7 * t ).xy;
    sig -= noise( 0.7 * t + 0.007 ).xy; // pseudo high pass. shoutouts to aaaidan
    return sig * decay;
}

vec2 snare( float t ) {
    float decay = exp( -t * 20.0 );
    vec2 snappy = 0.5 + 0.5 * noise( t ).xy;
    vec2 head = sin( TAU * ( t * 280.0 * vec2( 1.005, 0.995 ) - exp( -t * 100.0 ) ) );
  return clip( ( 3.0 * snappy * head ) * decay );
}

vec2 rimshot( float t ) {
    float attack = exp( -t * 400.0 ) * 0.6;
    vec2 wave = (
        tri( t * 450.0 * vec2( 1.005, 0.995 ) - attack ) +
        tri( t * 1800.0 * vec2( 0.995, 1.005 ) - attack )
    );
    return clip( 2.0 * wave * exp( -t * 300.0 ) );
}

vec2 tom( float t, float freq ) {
    float phase = freq * t - 5.0 * exp( -30.0 * t ) - 2.0 * exp( -100.0 * t );
    float decay = exp( -20.0 * t );
    return vec2( decay * sin( 2.0 * sin( TAU * phase ) ) );
}

// == main =========================================================================================
vec2 mainAudio( vec4 time ) {
  float beat = time.z * TIME2BEAT;

  vec2 dest = vec2( 0.0 );

  float tKick = euclideanRhythmsInteg(
    4.0,
    16.0,
    4.0 * beat
  ) / 4.0 * BEAT2TIME;
  dest += 0.5 * kick( tKick );

  float tHihat = euclideanRhythmsInteg(
    13.0,
    16.0,
    4.0 * beat
  ) / 4.0 * BEAT2TIME;
  dest += 0.2 * hihat( tHihat );

  float tSnare = euclideanRhythmsInteg(
    3.0,
    16.0,
    4.0 * beat - 4.0
  ) / 4.0 * BEAT2TIME;
  dest += 0.3 * snare( tSnare );

  float tHiTom = euclideanRhythmsInteg(
    3.0,
    10.0,
    4.0 * beat - 1.0
  ) / 4.0 * BEAT2TIME;
  dest += vec2( 0.2, 0.1 ) * tom( tHiTom, 180.0 );

  float tLoTom = euclideanRhythmsInteg(
    5.0,
    13.0,
    4.0 * beat - 2.0
  ) / 4.0 * BEAT2TIME;
  dest += vec2( 0.1, 0.2 ) * tom( tLoTom, 120.0 );

  float tRim = euclideanRhythmsInteg(
    3.0,
    5.0,
    4.0 * beat
  ) / 4.0 * BEAT2TIME;
  dest += 0.2 * rimshot( tRim );

  return clip( dest );
}
