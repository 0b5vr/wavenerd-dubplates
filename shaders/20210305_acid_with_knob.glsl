#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define beat *60.0/BPM

#define saturate(i) clamp(i, 0.,1.)
#define aSaturate(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define n2f(n) (440.0*pow(2.0,(n)/12.0))

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform vec4 param_knob2;
uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;

vec2 noise( float t ) {
  return sampleSinc( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) );
}

vec2 random2( float t ) {
  return fract( sampleNearest( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) ) );
}

vec2 wavetable( float freq, float time, float speed, float offset ) {
  if ( time < 0.0 ) { return vec2( 0.0 ); }
  float p = tri( freq * time );
  return noise( p * speed + offset );
}

vec2 kick( float t, float freq ) {
  float phase = freq * t - 4.0 * ( exp( -20.0 * t ) + exp( -80.0 * t ) + exp( -500.0 * t ) );
  float fmA = sin( TAU * 1.0 * phase + 1.4 );
  vec2 fmB = 0.5 * smoothstep( 0.25 beat, 2.0 beat, t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
  return aSaturate( 1.0 * vec2( exp( -1.0 * t ) * sin( TAU * phase + fmB ) ) );
}

vec2 clap( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  float amp = exp( -14.0 * t );
  amp *= mix(
    fract( 1.0 - 50.0 * t ),
    1.0,
    0.3 + 0.7 * smoothstep( 0.08, 0.09, t )
  );
  vec2 wave = wavetable( 4.0, t, 0.03, 0.0 );
  return lofir( amp * wave, 0.25 );
}

vec2 rimshot( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  float attack = exp( -t * 400.0 ) * 0.6;
  vec2 wave = (
    tri( t * 450.0 * vec2( 1.005, 0.995 ) - attack ) +
    tri( t * 1800.0 * vec2( 0.995, 1.005 ) - attack )
  );
  return aSaturate( 4.0 * wave * exp( -t * 400.0 ) );
}

vec2 filterSaw( float freq, float time, float cutoff, float resonance ) {
  if ( time < 0.0 ) { return vec2( 0.0 ); }
  vec2 sum = vec2( 0.0 );
  for ( int i = 1; i <= 32; i ++ ) {
    float fi = 0.5 + float( i );
    float cut = smoothstep( cutoff * 1.2, cutoff * 0.8, fi * freq );
    cut += smoothstep( cutoff * 0.3, 0.0, abs( cutoff - fi * freq ) ) * resonance;
    vec2 offset = vec2( -1.0, 1.0 ) * ( 0.1 * ( fi - 1.0 ) );
    sum += sin( fi * freq * time * TAU + offset ) / fi * cut;
  }
  return sum;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );
  
  float knob0 = paramFetch( param_knob0 );
  float knob1 = paramFetch( param_knob1 );
  float knob2 = paramFetch( param_knob2 );

  // -- kick ---------------------------------------------------------------------------------------
  float kickTime = mod( mod( time.y, 2.5 beat ), 1.5 beat );
  float sidechain = linearstep( 0.0, 0.6 beat, kickTime );
  {
    dest += 0.5 * kick( kickTime, n2f( -36.0 ) );
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t = mod( time.z, 0.25 beat );
    float begin = time.z - t;
    vec2 dice = random2( begin );
    float open = mix( 30.0, 200.0, dice.x );
    float amp = 0.3 * exp( -open * t );
    vec2 wave = wavetable( 20.0, lofi( t, 0.00005 ), 4.0, 0.0 );
    dest += amp * wave;
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t = mod( mod( time.y, 1.25 beat ), 0.75 beat );
    float amp = 0.05 * exp( -10.0 * t );
    vec2 wave = wavetable( 80.0, lofi( t, 0.00002 ), 0.01, 0.0 );
    dest += amp * wave;
  }

  // -- clap ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y - 1.0 beat, 2.0 beat );
    float amp = 0.3;
    dest += amp * clap( t );
  }

  // -- rimshot ------------------------------------------------------------------------------------
  {
    vec2 t = vec2(
      mod( mod( time.z, 2.25 beat ), 1.25 beat ),
      mod( mod( time.z - 0.25 beat, 2.75 beat ), 0.75 beat )
    );
    dest.x += 0.3 * rimshot( t.x ).x;
    dest.y += 0.3 * rimshot( t.y ).y;
  }

  // -- acid ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y, 0.25 beat );
    vec2 dice = random2( 0.94 * lofi( time.y, 0.25 beat ) );
    float filt = (
      100.0 +
      knob2 * mix( 1000.0, 4000.0, dice.x ) * exp( -mix( 20.0, 40.0, dice.y ) * t ) +
      knob0 * 4000.0
    );

    float pattern[16] = float[](
      -1.0, 0.0, 11.0, 0.0,
      0.0, 12.0, 1.0, 0.0,
      0.0, 6.0, 1.0, 0.0,
      -2.0, 0.0, 0.0, 4.0
    );
    float freq = n2f( -24.0 + pattern[ int( time.y / ( 0.25 beat ) ) ] );
    float amp = 0.25;
    dest += amp * aSaturate( 1.0 * filterSaw( freq, t, filt, filt / 100.0 * knob1 ) );
  }

  return aSaturate( dest );
}
