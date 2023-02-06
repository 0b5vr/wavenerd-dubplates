#define BPM bpm

#define PI 3.14159265359
#define TAU 6.28318530718
#define B2T (60./bpm)
#define T2B (bpm/60.)
#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

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

  float sidechain = 1.0;

  // kick
  {
    float t = time.x;
    sidechain = smoothstep( 1.0 * B2T, 0.999 * B2T, t ) * smoothstep( 0.0, 0.8 * B2T, t );

    {
      float env = linearstep( 0.3, 0.1, t );

      dest += 0.5 * env * tanh( 2.0 * sin(
        + 300.0 * t
        - 55.0 * exp( -35.0 * t )
        - 30.0 * exp( -500.0 * t )
      ) );
    }
  }

  { // hihat
    float t = mod( time.x, 0.25 * B2T );
    float st = floor( time.y * T2B * 4.0 );
    float env = exp( -exp2( 6.0 - 2.0 * fract( .632 * st ) ) * t );
    vec2 wave = shotgun( 4000.0 * t, 2.0, 0.0 );
    dest += mix( 0.2, 1.0, sidechain ) * env * wave;
  }

  { // clap
    float t = mod( time.y - B2T, 2.0 * B2T );
    float env = mix(
      exp( -15.0 * t ),
      exp( -100.0 * mod( t, 0.02 ) ),
      smoothstep( 0.05, 0.02, t )
    );
    vec2 wave = shotgun( 400.0 * t - 10.0 * exp( -70.0 * t ), 2.0, 0.0 );
    dest += 0.6 * mix( 0.3, 1.0, sidechain ) * env * tanh( 20.0 * wave );
  }

  { // ride
    float t = mod( time.x, 0.5 * B2T );
    float env = exp( -5.0 * t );
    vec2 wave = shotgun( 5200.0 * t, 3.0, 0.0 );
    dest += 0.4 * mix( 0.2, 1.0, sidechain ) * env * wave;
  }

  { // bass
    float t = time.z;

    float freq = p2f( 36.0 );
    float phase = t * freq;
    vec2 wave = mix(
      vec2( sin( TAU * phase ) ),
      shotgun( phase, 2.0, 0.9 ),
      0.8
    );
    dest += sidechain * wave;
  }

  { // pad
    float t = time.z;
    const float[8] chords = float[](0.,7.,10.,14.,15.,19.,24.,29.);

    vec2 sum = vec2( 0.0 );

    for ( int i = 0; i < 8; i ++ ) {
      float note = 48.0 + chords[ i ];
      float freq = p2f( note );
      float phase = t * freq;
      vec2 wave = shotgun( phase, 2.0, 0.94 );
      sum += mix( 0.2, 1.0, sidechain ) * wave / 4.0;
    }

    dest += sum;
  }

  { // arp
    const float[8] chords = float[](0.,2.,3.,5.,7.,10.,12.,14.);

    vec2 sum = vec2( 0.0 );

    for ( int i = 0; i < 16; i ++ ) {
      float tp = time.z;
      float t = mod( tp - 0.25 * B2T * float( i ), 4.0 * B2T );
      float st = floor( ( tp - t ) / B2T / 0.25 + 0.5 );
      tp = t;
      t = mod( t, 0.75 * B2T );
      float delaydecay = exp( -floor( ( tp - t ) / B2T / 0.75 + 0.5 ) );
      float env = exp( -10.0 * t );
      float arpseed = mod( floor( 9.8 * st ), 16.0 );
      float note = 48.0 + chords[ int( arpseed ) % 8 ] + 12.0 * floor( arpseed / 8.0 );
      float freq = p2f( note );
      float phase = t * freq;
      vec2 wave = shotgun( phase, 4.0, 0.97 );
      sum += mix( 0.2, 1.0, sidechain ) * env * delaydecay * wave;
    }

    dest += 0.8 * sum;
  }

  return tanh( dest );
}
