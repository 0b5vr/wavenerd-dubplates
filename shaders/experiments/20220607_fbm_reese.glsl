#define BPM bpm
#define AMEN_BPM 170.0

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define fs(i) (fract(sin((i)*114.514)*1919.810))
#define PI 3.14159265359
#define TAU 6.28318530718

uniform sampler2D image_fbm;

float ar( float t, float l, float a, float r ) {
  return linearstep( 0.0, a, t ) * linearstep( l, l - r, t );
}

vec2 orbit( float t ) {
  return vec2( cos( TAU * t ), sin( TAU * t ) );
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  float freq = 40.0;

  float amp = ar( time.y, timeLength.y, 0.1, 0.1 );

  for ( int i = 0; i < 6; i ++ ) {
    float fi = float( i );
    float frequ = freq * mix( 0.996, 1.004, fs( fi ) );
    float offu = fs( fi + 1.0 );
    vec2 pan = mix( vec2( 0.0, 1.0 ), vec2( 1.0, 0.0 ), fi / 5.0 );

    float reese = sin( PI * time.y / timeLength.y );

    vec2 uv = vec2( 0.5 );
    uv += 0.1 * reese * orbit( frequ * time.y + offu ); // keynote
    uv += 0.05 * reese * reese * reese * orbit( 11.0 * frequ * time.y + offu ); // harmonic
    uv += 0.1 * texture( image_fbm, 2.0 * uv + 0.5 ).xx; // fbm mod
    dest += 1.5 * amp * pan * ( texture( image_fbm, uv ).xx * 2.0 - 1.0 ) / 6.0; // fbm osc
  }

  dest += 0.5 * amp * sin( TAU * freq * time.y );

  return clip( dest );
}
