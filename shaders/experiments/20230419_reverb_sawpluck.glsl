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

mat2 r2d( float t ) {
  float c = cos( t );
  float s = sin( t );

  return mat2( c, -s, s, c );
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

vec2 shotgun( float t, float spread, float snap ) {
  vec2 sum = vec2( 0.0 );

  for ( int i = 0; i < 64; i ++ ) {
    vec3 dice = pcg3df( vec3( i ) );

    float partial = exp2( spread * dice.x );
    partial = mix( partial, floor( partial + 0.5 ), snap );

    sum += vec2( sin( TAU * t * partial ) ) * r2d( TAU * dice.y );
  }

  return sum / 64.0;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  { // arp
    const int N_CHORD = 8;
    const int[] CHORDS = int[](0,5,7,10,12,19,22,26);

    vec2 sum = vec2( 0.0 );

    repeat(i,16){
      float tp = time.z;
      float t = mod( tp - 0.25 * B2T * float( i ), 4.0 * B2T );
      float st = floor( ( tp - t ) / B2T / 0.25 + 0.5 );

      float arpseed = float( N_CHORD ) * fract( 0.61 * st );
      float note = 48.0 + float( CHORDS[ int( arpseed ) % N_CHORD ] );
      float freq = p2f( note );
      float phase = t * freq;


      vec2 notesum = vec2( 0.0 );

      repeat(j,96){
        vec3 dice = pcg3df( vec3( j ) );

        float partial = 1.0 + float( j );

        { // dry
	      float env = exp( -20.0 * t );
          float wave = sin( TAU * t * freq * partial );
          notesum += vec2( wave ) / partial * env;
        }

        { // wet
          mat2 rot = r2d( PI * cheapnoise( float( j ) + 0.1 * time.z ).x );

          float env = smoothstep( 0.0, 0.01, t ) * exp( -4.0 * t );
          float wave = sin( TAU * ( t - 0.01 ) * freq * partial );
          notesum += 0.3 * vec2( wave ) / partial * env * rot;
        }
      }

      notesum /= 2.0;

      sum += notesum;
    }

    dest = 0.6 * sum;
  }

  return tanh( dest );
}
