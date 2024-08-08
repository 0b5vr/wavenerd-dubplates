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
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define fs(i) (fract(sin((i)*114.514)*1919.810))
#define inRange(a,b,x) ((a)<=(x)&&(x)<(b))

const float TRANSPOSE = 2.0;

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;
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

float chords[8] = float[](
  0.0, 0.0, 3.0, 3.0, 7.0, 7.0, 10.0, 14.0
);

vec2 noise( float t ) {
  return sampleSinc( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) );
}

vec2 random2( float t ) {
  return fract( sampleNearest( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) ) );
}

float euclideanRhythmsInteg(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((t-pulses)/pulses)+1.+fract(time);
}

float euclideanRhythmsRest(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((steps-t-1.)/pulses)+1.-fract(time);
}

vec2 pan(float x){
  return mix(vec2(2,0),vec2(0,2),x);
}

vec2 shotgun(float t,float spread,float snap){
  vec2 sum=vec2(0);
  for(int i=0;i<64;i++){
    float dice=fs(float(i));

    float partial=exp2(spread*dice);
    partial=mix(partial,floor(partial+.5),snap);

    vec2 pan=mix(vec2(2,0),vec2(0,2),fs(dice));
    sum+=sin(TAU*t*partial)*pan;
  }
  return sum/128.;
}

vec2 kick( float t, float freq ) {
  float phase = freq * t - 11.0 * ( exp( -25.0 * t ) + exp( -100.0 * t ) + exp( -700.0 * t ) );
  float fmA = sin( TAU * 1.0 * phase + 1.4 );
  vec2 fmB = 0.5 * exp( -20.0 * t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
  return clip( 1.0 * vec2( exp( -4.0 * t ) * sin( TAU * phase + fmB ) ) );
}

vec2 rimshot( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  float attack = exp( -t * 400.0 ) * 0.6;
  vec2 wave = (
    tri( t * 450.0 * vec2( 1.005, 0.995 ) - attack ) +
    tri( t * 1800.0 * vec2( 0.995, 1.005 ) - attack )
  );
  return clip( 4.0 * wave * exp( -t * 400.0 ) );
}

vec2 filterSaw(float freq,float t,float cutoff,float reso){
  vec2 sum=vec2(0);
  for(int i=1;i<=64;i++){
    float fi=float(i);
    float freqp=freq*fi;
    float omega=freqp/cutoff;
    float omegaSq=omega*omega;

    float a=4.0*reso+omegaSq*omegaSq-6.0*omegaSq+1.0;
    float b=4.0*omega*(omegaSq-1.0);
    float cut=1.0/sqrt(a*a+b*b);
    float offset=atan(a,b);

    sum+=0.66*sin(freqp*t*TAU-offset)/fi*cut;
  }
  return sum;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );
  
  // kick
  float kickTime = mod( time.x, 1.0 beat );
  float sidechain = linearstep( 0.0, 0.6 beat, kickTime );

  if ( inRange( 0.0 beat, 61.0 beat, time.z ) ) {
    dest += 0.5 * kick( kickTime, 50.0 );
  }

  // hihat
  {
    float t = mod( time.x, 0.25 beat );

    float vel = fract( floor( time.y / ( 0.25 beat ) ) * 0.62 + 0.67 );
    float amp = mix( 0.2, 0.3, vel );
    float decay = mix( 140.0, 10.0, vel );
    dest += amp*tanh(8.*shotgun(4000.*t,2.,.0))*exp( -t * decay );
  }

  // clap
  {
    float t=mod(time.y-1. beat,2. beat);

    float env=mix(
      exp(-30.*t),
      exp(-200.*mod(t,.013)),
      exp(-100.*max(0.,t-.02))
    );

    vec2 uv=orbit(89.*t)+40.*t;

    dest+=.23*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }
  
  // beep
  {
    float t=mod(time.x,.25 beat);
    float fm = 4.*sin(TAU*7800.*t)*exp(-20.*t);
    float wave = sin(TAU*1000.*t+fm);
    float st=mod(floor((time.z)/(.25 beat)),9.);
    
    float l=fract(st*.423+.5)*(.2 beat);
    float amp=linearstep(.0,-.001,t-l);

    dest += .13*wave * amp;
  }

  // fm perc
  {
    float t=lofi(mod(time.x,.25 beat),.0001);
    float st=mod(floor(time.z/(.25 beat)),32.);
    float fmfreq=exp2(9.+3.*fract(st*.622+.1));
    float decay=exp2(2.+5.*fract(st*.428));
    float fmamp=exp2(2.+4.*fract(st*.322+.8));
    float cfreq=300.;
    vec2 wave=sin(fmamp*sin(fmfreq*t)*exp(-decay*t)+cfreq*t+vec2(0,1));
    dest+=.05*mix(.5,1.,sidechain)*wave;
  }

  // clav
  {
    float t=euclideanRhythmsInteg(6.,15.,time.y/(.25 beat))*(.25 beat);
    dest+=.3*pan(.3)*exp(-t*200.)*sin(17000.*t);
  }

  // rim

  {
    float t=euclideanRhythmsInteg(5.,14.,time.y/(.25 beat)-3.)*(.25 beat);

    float env=exp(-300.*t);

    dest+=.3*env*pan(.7)*tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    ));
  }

  // pad
  {
    vec2 sum = vec2( 0.0 );
    float st=mod(floor((time.z)/(.25 beat)),32.);

    int pitchTable[8] = int[]( 0, 3, 7, 10, 12, 14, 19, 26 );

    for ( int i = 0; i < 48; i ++ ) {
      float fi = float( i );

      float t = time.z;

      float trans = mod( time.z, 16.0 beat ) < ( 12.0 beat ) ? 0.0 : 0.0;
      float freq = p2f( float( 48 + pitchTable[ i % 8 ] ) + TRANSPOSE + trans )
        * mix( 0.99, 1.01, fs( fi ) );
      float offu = fs( fi + 4.0 );
      vec2 pan = mix( vec2( 0.0, 1.0 ), vec2( 1.0, 0.0 ), fi / 47.0 );

      vec2 uv = vec2( 0.5 );
      uv += 0.1 * time.z;
      vec2 uv1 = uv + 0.18 * orbit( freq * t + offu );
      float partial=mod(floor(time.z/(.25 beat)),3.)*2.+1.;
      vec2 uv2 = uv + 0.18 * orbit( partial * freq * t + offu + 0.3 );
      float diff = texture( image_fbm, uv1 ).x - texture( image_fbm, uv2 ).x;

      float ampdecay=exp(0.+4.*fract(st*.456));
      float amp = 0.1 * exp(-ampdecay*mod(time.x, .25 beat));// * mix( 0.3, 1.0, sidechain );
      sum += amp * pan * diff; // fbm osc
    }

    dest += clip( sum );
  }

  return tanh(1.5*dest);
}
