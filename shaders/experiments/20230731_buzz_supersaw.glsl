// based on Au5's "Supersaw Killer" Tutorial
// https://www.youtube.com/watch?v=IhvO8grER5s

#define BPM bpm

#define PI 3.14159265359
#define TAU 6.28318530718
#define B2T (60./bpm)
#define T2B (bpm/60.)
#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define repeat(i,n) for(int i=0;i<(n);i++)

const uint UINT_MAX = 0xffffffffu;

uvec3 pcg3d( uvec3 v ) {
  v = v * 1145141919u + 1919810u;

  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;

  v ^= v >> 16u;

  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;

  return v;
}

vec3 pcg3df( vec3 v ) {
  uvec3 r = pcg3d( floatBitsToUint( v ) );
  return vec3( r ) / float( 0xffffffffu );
}

vec2 cheapnoise( float t ) {
  uvec3 s = uvec3( t * 256.0 );
  float p = fract( t * 256.0 );

  vec3 dice;
  vec2 v = vec2( 0.0 );

  dice = vec3( pcg3d( s ) ) / float( UINT_MAX ) - vec3( 0.5, 0.5, 0.0 );
  v += dice.xy * smoothstep( 1.0, 0.0, abs( p + dice.z ) );
  dice = vec3( pcg3d( s + 1u ) ) / float( UINT_MAX ) - vec3( 0.5, 0.5, 1.0 );
  v += dice.xy * smoothstep( 1.0, 0.0, abs( p + dice.z ) );
  dice = vec3( pcg3d( s + 2u ) ) / float( UINT_MAX ) - vec3( 0.5, 0.5, 2.0 );
  v += dice.xy * smoothstep( 1.0, 0.0, abs( p + dice.z ) );

  return 2.0 * v;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  { // lead
    float l = 2.0 * B2T;
    float t = time.y;
    float q = l - t;

    const int N_CHORD = 8;
    const int[] CHORDS = int[]( -24, -12, 0, 7, 12, 15, 19, 19 );

    vec2 sum = vec2( 0.0 );

    repeat( i, N_CHORD ) {
      vec3 dice = pcg3df( vec3( i ) );

      float pitch = 60.0 + float( CHORDS[ i % N_CHORD ] );
      float freq = p2f( pitch );
      vec2 phase = vec2( freq * t );

      phase -= 0.005 * freq * exp( -100.0 * t ); // attack pitch bend
      phase += 0.1 * cheapnoise( 0.001 * phase.x ); // buzz noise
      phase += dice.xy; // stereo phase

      float env = linearstep( 0.0, 0.001, t ) * linearstep( 0.0, 0.01, q );

      vec2 wave = 2.0 * ( fract( phase ) - 0.5 ) + sin( TAU * phase );

      sum += 0.3 * env * wave;
    }

  dest += 0.5 * tanh( sum );
  }

  return tanh( dest );
}
