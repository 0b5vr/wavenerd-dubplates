#define BPM bpm

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define beat *60.0/BPM
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define fs(i) (fract(sin((i)*114.514)*1919.810))
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define f2p(i) (12.*(log(i)-LN440)/LN2+69.)
#define PI 3.14159265359
#define TAU 6.28318530718

uniform sampler2D sample_808hh;
uniform vec4 sample_808hh_meta;
uniform sampler2D image_fbm;

float envA( float t, float a ) {
  return linearstep( 0.0, a, t );
}

float envAR( float t, float l, float a, float r ) {
  return envA( t, a ) * linearstep( l, l - r, t );
}

vec2 orbit( float t ) {
  return vec2( cos( TAU * t ), sin( TAU * t ) );
}

vec2 shotgun(float phase,float spread,float snap){
  vec2 sum=vec2(0);
  for(int i=0;i<64;i++){
    float fi=float(i);
    float dice=fs(fi);

    float partial=exp2(spread*dice);
    partial=mix(partial,floor(partial+.5),snap);

    vec2 pan=mix(vec2(2,0),vec2(0,2),fs(dice));
    sum+=sin(TAU*phase*partial+fi)*pan;
  }
  return sum/64.;
}

vec2 kick( float t ) {
  vec2 noise = vec2( texture( image_fbm, t * vec2( 97, 10 ) ).x - 0.5 ) * exp( -10.0 * t );
  vec2 tt = t + 0.002 * noise;
  vec2 a = exp( -5.0 * tt ) * sin( TAU * (
    30.0 * tt - 5.0 * ( exp( -5.0 * tt ) + exp( -60.0 * t ) )
  ) );
  return clip( 2.0 * a );
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );
  
  float sidechain;

  {
    float t = time.x;

    dest += 0.5 * kick( t );
    
    sidechain = 1.0 - envAR( t, 0.3, 0.001, 0.3 );
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),16.);

    float envptn=fract(.4+.63*st);
    float env=exp(-exp2(4.5+2.*envptn)*t);

    dest+=.2*sidechain*tanh(10.*shotgun(4000.*t,2.5,.0)*env);
  }

  // ride
  {
    float t=mod(time.x-.5 beat,1. beat);

    float env=exp(-5.*t);

    dest+=.06*sidechain*tanh(10.*shotgun(5000.*t,1.5,.0)*env);
  }

  {
    float t = mod( time.x, 0.25 beat );

    float vel = fract( floor( time.y / ( 0.25 beat ) ) * 0.62 + 0.67 );
    float amp = mix( 0.1, 0.14, vel );
    float decay = mix( 140.0, 40.0, vel );
    dest += amp * sampleSinc( sample_808hh, sample_808hh_meta, t ) * exp( -t * decay );
  }

  {
    float l = 2.0 beat;
    float t = mod( time.y, l );
    float progress = t / l;
    float freq = p2f( 28.0 );

    float reese = mix( 0.1, 1.0, linearstep( 0.0 beat, 2.0 beat, t ) );

    vec2 sum = vec2( 0.0 );

    for ( int i = 0; i < 9; i ++ ) {
      float fi = float( i );
      float frequ = freq * mix( 0.996, 1.004, fs( fi ) );
      float offu = fs( fi + 4.0 );
      vec2 pan = mix( vec2( 0.0, 1.0 ), vec2( 1.0, 0.0 ), fi / 8.0 );

      vec2 uv = vec2( 0.5 );
      uv += 0.1 * reese * orbit( frequ * t + offu ); // keynote
      uv += 0.05 * reese * reese * orbit( 9.0 * frequ * t + offu ); // harmonic
      uv += 0.2 * progress * texture( image_fbm, 2.0 * uv + 0.5 ).xx; // fbm mod
      sum += pan * ( texture( image_fbm, uv ).xx * 2.0 - 1.0 ) / 9.0; // fbm osc
    }

    float amp = 0.6 * sidechain * envAR( t, l, 0.02, 0.1 );
    dest += amp * mix(
      clip( 8.0 * sum ),
      vec2( sin( TAU * freq * t ) ),
      0.5
    );
  }

  {
    vec2 sum = vec2( 0.0 );
    
    int pitchTable[8] = int[]( 0, 5, 7, 10, 12, 15, 19, 26 );

    for ( int i = 0; i < 48; i ++ ) {
      float fi = float( i );

      float l = 1.0 beat;
      float t = mod( time.z, l );
      float progress = t / l;

      float freq = p2f( float( 28 + 24 + pitchTable[ i % 8 ] ) ) * mix( 0.99, 1.01, fs( fi ) );
      float offu = fs( fi + 4.0 );
      vec2 pan = mix( vec2( 0.0, 1.0 ), vec2( 1.0, 0.0 ), fi / 47.0 );

      float offp = pow( 0.5, 1.0 + 3.0 * fs( lofi( time.z, 1.0 beat ) ) + fs( fi + 0.3 ) );
      vec2 uv = vec2( 0.5 );
      uv += 0.3 * time.z;
      vec2 uv1 = uv + 0.2 * orbit( freq * t + offu );
      vec2 uv2 = uv + 0.2 * orbit( freq * t + offp + offu );
      float diff = texture( image_fbm, uv1 ).x - texture( image_fbm, uv2 ).x;

      float amp = 0.1 * mix( 0.3, 1.0, sidechain ) * envAR( t, l, 0.02, 0.1 );
      sum += amp * pan * diff; // fbm osc
    }

    dest += clip( 2.0 * sum );
  }

  return clip( dest );
}
