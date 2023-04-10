#define BPM bpm
#define BEAT 60.0/BPM

#define saturate(i) clamp(i, 0.,1.)
#define aSaturate(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define n2f(n) (440.0*pow(2.0,((n)+TRANSPOSE)/12.0))
#define fs(s) (fract(sin((s)*114.514)*1919.810))
#define inRange(a,b,x) ((a)<=(x)&&(x)<(b))

const float TRANSPOSE = 1.0;

const float PI = 3.141592654;
const float TAU = 6.283185307;
const float P4 = 1.33483985417;
const float P5 = 1.49830707688;

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;

float chords[16] = float[](
  0.0, 3.0, 5.0, 7.0, 10.0, 12.0, 14.0, 15.0,
  -5.0, -3.0, -2.0, 2.0, 5.0, 9.0, 10.0, 12.0
);

mat3 getOrthBasis( vec3 z ) {
  z = normalize( z );
  vec3 x = normalize( cross( vec3( 0.0, 1.0, 0.0 ), z ) );
  vec3 y = cross( z, x );
  return mat3( x, y, z );
}

vec3 cyclicNoise( vec3 p ) {
  vec3 sum = vec3( 0.0 );
  float amp = 0.5;
  float warp = 1.4;
  mat3 rot = getOrthBasis( vec3( 0.8, -0.5, 0.2 ) );

  for ( int i = 0; i < 8; i ++ ) {
    p *= rot;
    p += sin( p.zxy * warp );
    sum += sin( cross( cos( p ), sin( p.yzx ) ) ) * amp;
    amp *= 0.5;
    warp *= 1.3;
    p *= 2.0;
  }

  return sum;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  // -- pads ---------------------------------------------------------------------------------------
  {
    vec3 wave = vec3( 0.0 );
    int iProg = int( time.z / ( 8.0 * BEAT ) ) % 2;
    for ( int i = 0; i < 128; i ++ ) {
      vec4 diceOsc = fs( vec4( 2, 3, 5, 7 ) * 10.68 * float( i ) );
      float t = time.z + TAU * diceOsc.z;
      float freq = n2f( chords[ ( i % 8 ) + 8 * iProg ] + 12.0 * floor( diceOsc.x * 3.0 - 2.0 ) );
      t += 0.01 * sin( t );
      float thetaFm = 3.0 * t * freq * TAU;
      vec2 circleFm = 2.0 * vec2( sin( thetaFm ), cos( thetaFm ) );
      vec3 posFm = 0.5 * cyclicNoise( vec3( circleFm, -4.0 + sin( t ) ) );
      float theta = TAU * t * freq;
      vec2 circle = 2.0 * vec2( sin( theta ), cos( theta ) );
      wave += vec3( cyclicNoise( posFm + vec3( circle, 8.0 + 4.0 * sin( t ) ) ).xy, 1.0 );
    }
    float amp = 2.0;
    dest += amp * wave.xy / wave.z;
  }

  return aSaturate( 1.2 * dest );
}
