#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define P4 1.33483985417
#define P5 1.49830707688
#define beat *60.0/BPM

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define st(i) pow(2.,(i)/12.)
#define p2f(i) (440.*st(i-69.))
#define fs(i) (fract(sin((i)*114.514)*1919.810))
#define inRange(a,b,x) ((a)<=(x)&&(x)<(b))

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;
uniform sampler2D sample_clapnoise;
uniform vec4 sample_clapnoise_meta;
uniform sampler2D sample_snarenoise;
uniform vec4 sample_snarenoise_meta;
uniform sampler2D sample_808hh;
uniform vec4 sample_808hh_meta;
uniform sampler2D image_fbm;

vec2 random2( float t ) {
  return fract( sampleNearest( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) ) );
}

float envA( float t, float a ) {
  return linearstep( 0.0, a, t );
}

float envAR( float t, float l, float a, float r ) {
  return envA( t, a ) * linearstep( l, l - r, t );
}

float asymsine( float p, float dist ) {
  return sin( TAU * p + dist * cos( TAU * p ) );
}

vec2 asymsine( vec2 p, float dist ) {
  return sin( TAU * p + dist * cos( TAU * p ) );
}

vec2 orbit( float t ) {
  return vec2( cos( TAU * t ), sin( TAU * t ) );
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  int progression[24] = int[](
    39, 53, 54, 56, 58, 61,
    44, 51, 54, 58, 59, 61,
    41, 55, 56, 58, 60, 63,
    34, 50, 54, 56, 58, 61
  );
  int progress = 6 * ( int( time.z / ( 8.0 beat ) ) % 4 );

  // -- kick ---------------------------------------------------------------------------------------
  float sidechain;

  {
  	float t = time.x;
    sidechain = mix( 0.3, 1.0, linearstep( 0.0, 1.0 beat, t ) );

    if ( time.z < 61.0 beat ) {
      float phase = 300.0 * t - 70.0 * exp( -30.0 * t );
      dest += 0.5 * tanh( 1.5 * sin( phase ) ) * exp( -6.0 * t );
    }
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t = mod( mod( time.x, 0.5 beat ), 0.3 beat );

    float s = floor( ( time.z ) / ( 0.5 beat ) );
    s += floor( ( time.z + 0.2 beat ) / ( 0.5 beat ) );
    s = mod( s, 16.0 );

    float vel = fract( s * 0.62 + 0.67 );
    float amp = mix( 0.14, 0.18, vel );
    float decay = mix( 70.0, 5.0, vel );
    dest += amp * sidechain * sampleSinc( sample_808hh, sample_808hh_meta, t ) * exp( -t * decay );
  }

  // -- ride ---------------------------------------------------------------------------------------
  {
    float t = mod( time.x - 0.5 beat, 1.0 beat );

    vec2 wave = vec2( asymsine( 5103.0 * t, 1.0 ) ) * exp( -3.0 * t );
    wave = asymsine( 14915.0 * t + wave, 1.0 );
    wave += sampleSinc( sample_snarenoise, sample_snarenoise_meta, t );

    dest += 0.08 * sidechain * wave * exp( -3.0 * t );
  }

  // -- clap ---------------------------------------------------------------------------------------
  {
    float t = mod( time.y + 1.0 beat, 2.0 beat );

    float env = mix(
      exp( -t * 40.0 ),
      exp( -mod( t, 0.014 ) * 300.0 ),
      exp( -t * 50.0 )
    );

    dest += sidechain * env * tanh( 5.0 * sampleSinc( sample_clapnoise, sample_clapnoise_meta, t ) );
  }

  // -- snare --------------------------------------------------------------------------------------
  if ( time.z > 60.0 beat ) {
    float t = (
      time.z < 62.0 beat ? mod( mod( time.x, 0.5 beat ), 0.3 beat ) :
      mod( time.x, 0.125 beat )
  	);

    vec2 wave = sampleSinc( sample_snarenoise, sample_snarenoise_meta, t );
    wave += asymsine( t * 240.0 - 4.0 * exp( -300.0 * t ), 0.5 );
    wave -= 0.5 * asymsine( t * 240.0 * 1.5 - 4.0 * exp( -300.0 * t ), 0.5 );

    dest += 0.3 * sidechain * exp( -20.0 * t ) * tanh( 1.0 * wave );
  }

  // -- bass ---------------------------------------------------------------------------------------
  {
    vec2 sum = vec2( 0.0 );

    float t = mod( time.y, 2.0 beat );
    float t0 = (
      t < ( 0.8 beat ) ? 0.0 beat :
      t < ( 1.3 beat ) ? 0.8 beat :
      t < ( 1.5 beat ) ? 1.3 beat :
      1.5 beat
    );
    float poff = (
      t < ( 1.3 beat ) ? 0.0 :
      t < ( 1.5 beat ) ? 5.0 :
      7.0
    );
    t -= t0;

    int note = progression[ progress ];
    float freq = p2f( mod( float( note ) + poff + 6.0, 12.0 ) + 30.0 );
    float cutoff = p2f( 40.0 + 80.0 * exp( -5.0 * t ) );
    float reso = 0.5;

    for ( int i = 1; i <= 192; i += 2 ) {
      float fi = float( i );
      float freqp = freq * fi;
      float omega = freqp / cutoff;
      float omegaSq = omega * omega;

      float a = 4.0 * reso + omegaSq * omegaSq - 6.0 * omegaSq + 1.0;
      float b = 4.0 * omega * ( omegaSq - 1.0 );
      float coeff = 1.0 / sqrt( a * a + b * b );
      float phase = atan( a, b );

      sum += sin( freqp * t * TAU - phase ) / fi * coeff;
    }

    dest += 0.4 * sidechain * sum * linearstep( 0.0, 0.001, t );
  }

  // -- pad ----------------------------------------------------------------------------------------
  {
    vec2 sum = vec2( 0.0 );

    float l = 8.0 beat;
    float t = mod( time.z, l );

    for ( int i = 0; i < 48; i ++ ) {
      float fi = float( i );

      int note = progression[ i % 6 + progress ];
      float freq = p2f( float( 12 + note ) ) * mix( 0.99, 1.01, fs( fi ) );
      float offu = fs( fi + 4.0 );
      vec2 pan = mix( vec2( 0.0, 1.0 ), vec2( 1.0, 0.0 ), fi / 47.0 );

      float offp = 0.2;
      vec2 uv = vec2( 0.0 );
      uv += 0.01 * t;
      vec2 uv1 = uv + vec2( 0.2, 0.05 ) * orbit( freq * t + offu );
      vec2 uv2 = uv + vec2( 0.2, 0.05 ) * orbit( freq * t + offp + offu );
      float diff = texture( image_fbm, uv1 ).x - texture( image_fbm, uv2 ).x;

      float amp = envAR( t, l, 0.02, 0.1 );
      sum += amp * sidechain * pan * diff; // fbm osc
    }

    dest += 0.07 * sum;
  }

  // -- arp ----------------------------------------------------------------------------------------
  {
    vec2 wave = vec2( 0.0 );

    for ( int i = 0; i < 4; i ++ ) {
      float fi = float( i );
      float t = mod( mod( time.x, 0.5 beat ), 0.3 beat );

      float s = floor( ( time.z - fi beat ) / ( 0.5 beat ) );
      s += floor( ( time.z - fi beat + 0.2 beat ) / ( 0.5 beat ) );
      s = mod( s, 256.0 );

      vec2 dice = random2( 0.81 * fi );

      float arpseed = fract( 0.61 * s );
      float note = float( progression[ int( arpseed * 18.0 ) % 6 + progress ] ) + 12.0 * floor( arpseed * 3.0 );
      float freq = p2f( note );

      float env = exp( -t * 10.0 );
      vec2 amp = saturate( 0.5 + vec2( 0.5, -0.5 ) * sin( 2.0 * fi + time.w ) * fi ) * exp( -fi ) * env;

      vec2 phase = t * freq + mix( vec2( 1.0, 0.0 ), vec2( 0.0, 1.0 ), dice.y );

      // fm e piano
      float tauPhase = TAU * t * freq;
      float fbPhase = TAU * pow( fract( st( 0.07 ) * t * freq ), 0.7 ); // cringe pseudo feedback

      wave += amp * (
        + exp( -1.0 * t ) * cos( st( 0.03 ) * tauPhase + exp( -8.0 * t ) * cos( 14.0 * tauPhase ) ) // OP1, OP2
        + exp( -1.0 * t ) * cos( tauPhase + 3.0 * exp( -1.0 * t ) * cos( 1.0 * tauPhase ) ) // OP3, OP4
        + exp( -1.0 * t ) * cos( st( -0.07 ) * tauPhase + 3.0 * exp( -2.0 * t ) * cos( fbPhase ) ) // OP5, OP6
      );
    }

    dest += 0.12 * sidechain * wave;
  }

  return tanh( 2.0 * dest );
}
