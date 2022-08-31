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
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define f2p(i) (12.*(log(i)-LN440)/LN2+69.)
#define fs(i) (fract(sin((i)*114.514)*1919.810))

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform vec4 param_knob2;
uniform vec4 param_knob3;

uniform sampler2D image_fbm;

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

  float sidechain=mix(1.,.3,smoothstep(1. beat,.0,time.x)*linearstep(.0,.001,time.x));

  // kick
  {
    float t=time.x;

    float env=linearstep(.0,.001,t)*linearstep(0.3,0.1,t);

    dest+=.5*tanh((
      env*tanh(1.5*sin(360.*t-35.*exp(-35.*t)))
      +tanh(1.5*sin(TAU*3.*exp(-500.*t)))
    ));
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);

    float env=exp(-50.*t);
    env *= linearstep(.0,.001,t);

    vec2 uv=orbit(1100.*t)+100.*t;

    dest+=.4*sidechain*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5);
  }

  // hihat
  {
    float t=mod(time.x-.5 beat,1. beat);

    float env=exp(-15.*t);

    vec2 uv=orbit(800.*t)+137.*t;

    dest+=.2*sidechain*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5);

    uv=orbit(802.*t)+137.*t;

    dest-=.2*sidechain*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5);
  }

  // clap
  {
    float t=mod(time.y-1. beat,2. beat);

    float env=mix(
      exp(-40.*t),
      exp(-200.*mod(t,.013)),
      exp(-100.*max(0.,t-.02))
    );

    vec2 uv=orbit(79.*t)+25.*t;

    dest+=0.2*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }

  // tom
  {
    float t=mod(time.y-.25 beat,2. beat);

    float env=exp(-20.*t);
    env*=linearstep(.0,.001,t);

    dest+=vec2(.2,.1)*env*sin(3.*sin(1100.*t-10.*exp(-100.*t)));
  }

  // tom
  {
    float t=mod(time.y-1.5 beat,2. beat);

    float env=exp(-20.*t);
    env *= linearstep(.0,.001,t);

    dest+=vec2(.1,.2)*env*sin(3.*sin(800.*t-10.*exp(-100.*t)));
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

  // acid
  {
    float t=mod(time.y,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),9.);

    float env=envAR(t,.25 beat,.001,.01);
    float cutoff=p2f(40.+40.*exp(-10.*t));
    vec2 wave=filterSaw(p2f(32.),t,cutoff,.0);

    dest+=.2*env*sidechain*wave;
  }

  // acid
  {
    float t=mod(time.y,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),9.);

    int arp[9]=int[](0,-12,-7,12,0,17,-12,13,12);
    float l=.25 beat*min(1.,mix(.5,1.2,fract(.62*st)));

    float freq=p2f(44.+float(arp[int(st)]));
    float env=envAR(t,l,.001,.01);
    float cutoff=p2f(96.*mix(exp(-10.*t),1.,.7)+12.*sin(time.w));
    vec2 wave=filterSaw(freq,t,cutoff,.8);

    dest+=.13*env*sidechain*tanh(20.*wave);
  }

  return tanh(2.*dest);
}
