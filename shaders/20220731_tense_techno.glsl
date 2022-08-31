#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define TIME2BEAT (BPM/60.)
#define beat *60.0/BPM

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define pan(p) (1.+vec2(-(p),(p)))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define f2p(i) (12.*(log(i)-LN440)/LN2+69.)
#define fs(i) (fract(sin((i)*114.514)*1919.810))

uniform sampler2D image_fbm;

float seed;

float random() {
  seed=fs(seed);
  return 2.*seed-1.;
}

float sinc(float x){
  return x==0.?1.:sin(TAU*x)/TAU/x;
}

vec2 orbit(float t){
  return vec2(cos(TAU*t), sin(TAU*t));
}

vec2 distort( vec2 x, float p ) {
  return sign( x ) * pow( abs( x ), vec2( p ) );
}

float envA( float t, float a ) {
  return linearstep( 0.0, a, t );
}

float envAR( float t, float l, float a, float r ) {
  return envA( t, a ) * linearstep( l, l - r, t );
}

float euclideanRhythmsInteg(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((t-pulses)/pulses)+1.+fract(time);
}

vec2 filterSaw(float freq,float t,float cutoff,float reso){
  vec2 sum = vec2(0);
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

vec2 mainAudio(vec4 time){
  vec2 dest = vec2(0);

  float kickt=mod(mod(time.y,2. beat),.75 beat);
  float sidechain=mix(1.,.3,smoothstep(1. beat,0.,kickt));

  // kick
  {
    float t=kickt;
    float env=linearstep(.0,.001,t)*linearstep(0.3,0.1,t);

    dest+=.5*tanh((
      env*tanh(1.5*sin(360.*t-65.*exp(-80.*t)))
    ));
  }

  // bass
  {
    float t=kickt;

    vec2 uv=orbit(4.2*t)+t;
    dest+=.5*sidechain*texture(image_fbm,uv).x;
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);

    float env=exp(-50.*t);
    env *= linearstep(.0,.001,t);

    vec2 uv=orbit(1100.*t)+100.*t;

    dest+=.3*sidechain*env*tanh(5.*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5));
  }

  // clap
  {
    float t=mod(time.y-3. beat,4. beat);

    float env=exp(-40.*t)+.02*exp(-5.*t);

    t+=0.1*sin(t*90.0);
    vec2 uv=orbit(59.8*t)+25.*t;

    dest+=.2*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }

  // rim
  {
    float t=euclideanRhythmsInteg(10.,16.,time.y*4.*TIME2BEAT-2.)*.25 beat;

    float env=exp(-300.*t);

    dest+=.2*env*tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    ));
  }

  // noise
  {
    float t=mod(time.x,.25 beat);
    float st=floor(time.z/(.25 beat));
    vec2 uv=exp(1.-4.*fs(st+.4))*orbit(exp(3.+3.*fs(st))*t);

    float env=exp(-100.0*fract(mod(st,32.)*.631+.5)*t);

    dest+=.2*sidechain*env*tanh(5.*(texture(image_fbm,uv).x-.5))*pan(fs(st+.4)-.5);
  }

  {
    seed = 0.199;

    for ( int i = 0; i < 50; i ++ ) {
      float reltone = 1.0 + random() * 4.0;

      float relfreq = pow( 2.0, reltone );
      float relfreqOt = floor( relfreq + 0.5 );
      float relfreqH = mix( relfreq, relfreqOt, 0.2 );
      reltone = log2( relfreqH ) * 12.0;

      float mtone = reltone;
      float mfreq = 220.0 * pow( 2.0, mtone / 12.0 );

      for ( int j = 0; j < 50; j ++ ) {
        float ptone = mtone + random() * 0.5;

        float freq = 220.0 * pow( 2.0, ptone / 12.0 );

        float noisePhase = TAU * fract( freq * time.z * 5.0 );
        float tt = time.z + pow( time.z / 10.0, 2.0 ) + 0.0001 * sin( TAU / 32.0 * noisePhase );

        vec2 phase = TAU * fract( freq * tt ) + TAU * vec2( random(), random() );
        dest += 0.001*sidechain*sin( phase );
      }
    }
  }

  return tanh(2.*dest);
}
