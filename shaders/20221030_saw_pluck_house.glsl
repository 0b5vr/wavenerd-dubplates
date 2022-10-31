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
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define f2p(i) (12.*(log(i)-LN440)/LN2+69.)
#define fs(i) (fract(sin((i)*114.514)*1919.810))
#define inrange(x,a,b) ((a)<=(x)&&(x)<(b))

const float TRANSPOSE=0.;

const float PI=acos(-1.);
const float TAU=2.*PI;
const float LN2=log(2.);

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform vec4 param_knob2;
uniform vec4 param_knob3;

uniform sampler2D image_fbm;

uvec3 pcg3d(uvec3 v){
  v=v*1145141919u+1919810u;
  v.xyz+=v.yzx*v.zxy;
  v^=v>>16u;
  v.xyz+=v.yzx*v.zxy;
  return v;
}

mat2 r2d(float x){
  float c=cos(x),s=sin(x);
  return mat2(c, s, -s, c);
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

float euclideanRhythmsRest(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((steps-t-1.)/pulses)+1.-fract(time);
}

vec2 shotgun(float phase,float spread,float snap){
  vec2 sum=vec2(0);
  for(int i=0;i<64;i++){
    float fi=float(i);
    float dice=fs(fi);

    float partial=exp2(spread*dice);
    partial=mix(partial,floor(partial+.5),snap);

    sum+=r2d(fi)*vec2(sin(TAU*phase*partial+fi));
  }
  return sum/64.;
}

vec2 filterSaw(float freq,vec2 t,float cutoff,float reso){
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

vec2 filterSaw(float freq,float t,float cutoff,float reso){
  return filterSaw(freq,vec2(t),cutoff,reso);
}

float glidephase(float t,float t1,float p0,float p1){
  float m0=(p0-69.)/12.;
  float m1=(p1-69.)/12.;
  float b=(m1-m0)/t1;

  return (
    +p2f(p0)*(exp2(b*min(t,t1))-1.)/b/LN2
    +max(0.,t-t1)*p2f(p1)
  );
}

vec2 mainAudio(vec4 time){
  vec2 dest=vec2(0);

  // time=lofi(time,4. beat)+mod(time,1. beat);
  // time=lofi(time,1. beat)+mod(time,.166 beat);
  // time=lofi(time,1. beat)+mod(time,.25 beat);
  // time=mix(
  //   time,
  //   lofi(time,1. beat)+mix(mod(time,1. beat),mod(time,.03),.8),
  //   step(3. beat,time.y)
  // );
  // time+=mix(
  //   0.,
  //   -3.*(1.-exp(-4.*time.x)),
  //   step(3. beat,time.y)
  // );
  time=mod(time,timeLength);

  float kickt;
  float sidechain;

  // kick
  {
    float t=mod(time.x,1. beat);
    // t=euclideanRhythmsInteg(7.,16.,time.z*4.*TIME2BEAT-2.)/(4.*TIME2BEAT);
    sidechain=1.-smoothstep(1. beat,.0,t)*linearstep(.0,.001,time.x);

    if(inrange(time.z,0. beat,61. beat)){
      float env=linearstep(.0,.001,t)*linearstep(0.3,0.1,t);

      dest+=.5*tanh((
        env*tanh(1.5*sin(360.*t-45.*exp(-35.*t)))
        +tanh(1.5*sin(TAU*3.*exp(-500.*t)))
      ));
    }
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),16.);

    float env=exp(-exp2(4.5+3.*fract(.5+.642*st))*t);

    dest+=.2*tanh(10.*shotgun(4000.*t,2.5,.0)*env);
  }

  // clap
  {
    float t=mod(time.y-1. beat,2. beat);

    float env=mix(
      exp(-30.*t),
      exp(-200.*mod(t,.01)),
      exp(-100.*max(0.,t-.02))
    );

    vec2 uv=orbit(69.*t)+41.*t;

    dest+=.3*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }

  // clav
  {
    float t=euclideanRhythmsInteg(3.,5.,time.y/(.25 beat))*(.25 beat);
    dest+=.2*r2d(-.7)*exp(-t*200.)*vec2(sin(17000.*t));
  }

  // rim
  {
    float t=euclideanRhythmsInteg(9.,14.,time.y/(.25 beat)-3.)*(.25 beat);

    float env=exp(-300.*t);

    dest+=.2*env*r2d(.7)*vec2(tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    )));
  }

  // cowbell
  {
    vec2 t=mod(time.y-2.25 beat,4. beat)*vec2(.99,1.0);

    vec2 cow=tri(t*540.54)+tri(t*800.);

    dest+=.2*tanh(2.*cow)*exp(-t*20.0);
  }

  // ride
  {
    float t=mod(time.x-.5 beat,1. beat);

    float env=exp(-5.*t);

    dest+=.06*sidechain*tanh(10.*shotgun(6000.*t,1.,.0)*env);
  }

  // crash
  {
    float t=time.z;

    float env=mix(exp(-1.*t),exp(-5.*t),.8);

    dest+=.2*mix(.5,1.,sidechain)*tanh(10.*shotgun(3800.*t,1.9,.2)*env);
  }

  // bass
  {
    float stf=mod(time.y/(.25 beat),16.);
    float t=euclideanRhythmsInteg(5.,13.,stf+2.)*(.25 beat);
    float rest=euclideanRhythmsRest(5.,13.,stf+2.)*(.25 beat);
    float st=mod(floor(stf-t/(.25 beat)+.5),4.);

    float freq=p2f(24.+TRANSPOSE);
    float env=linearstep(0.,.001,t)*linearstep(0.,.01,rest);
    env*=linearstep(0.,.001,time.y)*linearstep(0.,.01,4. beat-time.y);
    env*=exp(-t*exp(2.*fract(st*.388+.6)));
    float cutoff=p2f(80.*mix(exp(-10.*t),1.,.8));

    float fm=env*sin(3.*TAU*freq*t);
    float osc=sin(TAU*freq*t+2.*fm);

    dest+=.5*sidechain*env*tanh(2.*osc);
  }

  // chord
  {
    const int chord[8]=int[](0,7,10,12,14,15,26,29);

    vec2 sum=vec2(0);
    for(int iDelay=0;iDelay<3;iDelay++){
      float stf=mod(time.y/(.25 beat)-2.1*float(iDelay),16.);
      float t=euclideanRhythmsInteg(6.,15.,stf)*(.25 beat);
      float rest=euclideanRhythmsRest(6.,15.,stf)*(.25 beat);
      float st=mod(floor(stf-t/(.25 beat)+.5),4.);

      float filtenv=exp(-exp2(1.+2.*fract(st*.443))*t);
      float cutoff=p2f(120.*mix(filtenv,1.,.5)*exp(-.1*float(iDelay)));

      float env=linearstep(0.,.001,t)
        *linearstep(0.,.01,rest)
        *exp(-5.*t);
      mat2 delaypan=r2d(.5*float(iDelay)*sin(float(iDelay)+time.w));
      mat2 amp=env
        *delaypan
        *exp(-float(iDelay));

      for(int i=0;i<8;i++){
        float fi=float(i);
        mat2 chordpan=r2d(-4.+.8*mod(fi,2.));
        float freq=p2f(48.+TRANSPOSE+float(chord[i]))
          *(1.+.003*(fract(fi*.622)-.5));
        sum+=amp
          *chordpan
          *(1.-2.*mod(fi,2.))
          *filterSaw(freq,t+vec2(0,3)+fi,cutoff,.1);
      }
    }

    dest+=.2
      *mix(.5,1.,sidechain)
      *sum;
  }

  return tanh(1.5*dest);
}
