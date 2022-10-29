#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define TIME2BEAT (BPM/60.)
#define BEAT2TIME (60./BPM)
#define beat *BEAT2TIME
#define P4 1.33483985417
#define P5 1.49830707688

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

const float TRANSPOSE=6.;

uniform sampler2D image_fbm;

float sinc(float x){
  return x==0.?1.:sin(TAU*x)/TAU/x;
}

vec2 orbit(float t){
  return vec2(cos(TAU*t), sin(TAU*t));
}

float euclideanRhythmsInteg(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((t-pulses)/pulses)+1.+fract(time);
}

float euclideanRhythmsRest(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((steps-t-1.)/pulses)+1.-fract(time);
}

vec2 mainAudio(vec4 time){
  vec2 dest=vec2(0.);

  float sidechain=mix(1.,.3,smoothstep(BEAT2TIME,.0,time.x)*linearstep(.0,.001,time.x));

  // kick
  {
    float t=time.x;

    float env=exp(-20.*max(.0,t-.1));
    env*=linearstep(.0,.001,t);

    dest+=.4*env*tanh(1.5*sin(360.*t-35.*exp(-35.*t)));
    dest+=.2*tanh(1.5*sin(TAU*3.*exp(-500.*t)));
  }

  // bass
  {
    float t=time.x;

    vec2 uv=orbit(4.2*t)+t;
    dest+=.5*sidechain*mix(
      sin(TAU*p2f(24.+TRANSPOSE)*t),
      texture(image_fbm,uv).x,
      0.5
    );
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

    float env=exp(-10.*t);

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

    float env=exp(-10.*max(t-.1,0.));
    env*=1.+sinc(20.*t);

    vec2 uv=orbit(127.*t)+200.*t;

    dest+=0.3*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5);
  }

  // ride
  {
    float t=mod(time.x,.5 beat);

    float env=exp(-5.*t);
    env *= linearstep(.0,.001,t);

    vec2 uv=vec2(37.,73.)*t;
    uv=.7*vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )+.3*exp(-5.*t)*orbit(2200.0*t)+1500.0*t;

    dest+=.2*sidechain*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5);
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

  // chord
  {
    vec2 sum=vec2(0.);

    int pitchTable[3]=int[](0,3,7);

    for(int i=0;i<96;i++){
      float fi=float(i);

      float delay=floor(fs(fi+2.)*3.);
      float t=mod(time.y-.75 beat,2. beat)-delay*.75 beat;

      float freq=p2f(36.+TRANSPOSE+float(pitchTable[i%3]))*mix(.997,1.003,fs(fi));
      float offu=fs(fi+4.);
      vec2 pan=mix(vec2(0.,1.),vec2(1.,0.),fract(fi*.61));

      vec2 uv=vec2(.5);
      uv+=.3*exp(-4.*t)*(orbit(freq*t+offu)+.1*orbit(freq*13.*t+offu));
      float diff=texture(image_fbm,uv).x-.5;
      diff=fs(fi+7.)<.5?-diff:diff;

      float ampenv=exp(-3.*max(t-.1,0.));
      ampenv*=linearstep(.0,.001,t);
      float amp=.24*ampenv*sidechain*exp(-delay);
      sum+=amp*pan*diff;
    }

    dest+=sum;
  }

  return tanh(1.5*dest);
}
