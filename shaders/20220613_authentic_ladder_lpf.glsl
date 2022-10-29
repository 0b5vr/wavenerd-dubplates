#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define beat *60.0/BPM

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define f2p(i) (12.*(log(i)-LN440)/LN2+69.)

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform vec4 param_knob2;
uniform vec4 param_knob3;

vec2 distort( vec2 x, float p ) {
  return sign( x ) * pow( abs( x ), vec2( p ) );
}

float envA( float t, float a ) {
  return linearstep( 0.0, a, t );
}

float envAR( float t, float l, float a, float r ) {
  return envA( t, a ) * linearstep( l, l - r, t );
}

vec2 kick( float t, float freq ) {
  float phase = freq * t - 4.0 * ( exp( -20.0 * t ) + exp( -80.0 * t ) + exp( -500.0 * t ) );
  float fmA = sin( TAU * 1.0 * phase + 1.4 );
  vec2 fmB = 0.5 * smoothstep( 0.25 beat, 2.0 beat, t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
  return clip( 1.0 * vec2( exp( -1.0 * t ) * sin( TAU * phase + fmB ) ) ) * exp( -5.0 * t );
}

vec2 filterSaw( float freq, float time, float cutoff, float reso ) {
  if ( time < 0.0 ) { return vec2( 0.0 ); }
  vec2 sum = vec2( 0.0 );
  for ( int i = 1; i <= 192; i ++ ) {
    float fi = float( i );
    float freqp = freq * fi;
    float omega = freqp / cutoff;
    float omegaSq = omega * omega;

    // float R = 1.0 - reso;
    // float a = 1.0 - omega * omega;
    // float cut = 1.0 / sqrt( 4.0 * R * R * omega * omega + a * a );
    // float phase = atan( a, -2.0 * R * omega );

    float a = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;
    float b = 4.0 * omega * ( omegaSq - 1.0 );
    float cut = 1.0 / sqrt( a * a + b * b );
    float phase = atan( a, b );

    sum += 0.66 * sin( freqp * time * TAU - phase ) / fi * cut;
  }
  return sum;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );
 
  float knob0 = paramFetch( param_knob0 );
  float knob1 = paramFetch( param_knob1 );
  float knob2 = paramFetch( param_knob2 );
  float knob3 = paramFetch( param_knob3 );
  
  // -- kick ---------------------------------------------------------------------------------------
  float kickTime = time.x;
  {
    dest += 0.5 * kick( kickTime, 50.0 );
  }

  // -- acid ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y, 0.25 beat );
    float st = mod( floor( time.z / ( 0.25 beat ) ), 9.0 );

    float pitches[9] = float[](
      0.0, -12.0, -7.0, 12.0,
      0.0, 17.0, -12.0, 13.0,
      12.0
    );
    float l = 0.25 beat * min( 1.0, mix( 0.5, 1.2, fract( 0.62 * st ) ) );

    float freq = p2f( 48.0 + pitches[ int( st ) ] );
    float amp = 0.25 * envAR( t, l, 0.0, 0.01 );
    float cutoff = p2f( 96.0 * knob0 + 96.0 * knob1 * exp( -10.0 * t ) );
    vec2 wave = filterSaw( freq, t, cutoff, knob2 );

    dest += amp * distort( wave, 1.0 - knob3 );
  }

  return clip( dest );
}
