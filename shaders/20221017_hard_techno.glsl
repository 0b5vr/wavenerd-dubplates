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

const float TRANSPOSE=4.;

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

vec2 pan(float x){
  return mix(vec2(2,0),vec2(0,2),x);
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

    vec2 pan=mix(vec2(2,0),vec2(0,2),fs(dice));
    sum+=sin(TAU*phase*partial+fi)*pan;
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
    float t=kickt=time.x;
    sidechain=smoothstep(0.,.8 beat,t);

    if(inrange(time.z,0. beat,61. beat)){
      float env=linearstep(0.3,0.1,t);

      dest+=.5*env*tanh(2.*sin(
        340.*t-55.*exp(-35.*t)
        -30.*exp(-500.*t)
      ));
    }
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),16.);

    float envptn=fract(.4+.63*st);
    float env=exp(-exp2(4.5+2.*envptn)*t);

    dest+=.2*sidechain*tanh(10.*shotgun(4000.*t,2.5,.0)*env);
  }

  // clap
  {
    float t=mod(time.y-1. beat,2. beat);

    float env=mix(
      exp(-20.*t),
      exp(-200.*mod(t,.01)),
      exp(-100.*max(0.,t-.02))
    );

    vec2 uv=orbit(69.4*t)+40.*t;

    dest+=.25*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }

  // shaker
  {
    float t=mod(time.y,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),4.);
    vec3 dice=vec3(pcg3d(uvec3(st+40.)))/float(0xffffffffu);

    float env=exp(-10.*t);

    float m=exp(1.+2.*dice.y);
    vec2 uv=orbit(m*72.*t)+m*55.7*t;

    dest+=.1*mix(.1,1.,sidechain)*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.1).x
    )-.5));
  }

  // fm perc
  {
    float t=lofi(mod(time.x,.25 beat),8E-5);
    float st=mod(floor(time.z/(.25 beat)),8.);
    float fmfreq=exp2(10.+3.*fract(st*.622+.1));
    float decay=exp2(5.*fract(st*.428));
    float fmamp=exp2(2.+4.*fract(st*.322+.8));
    float cfreq=300.;
    vec2 wave=sin(fmamp*sin(fmfreq*t)*exp(-decay*t)+cfreq*t+vec2(0,2));
    dest+=.13*mix(.5,1.,sidechain)*wave;
  }

  // clav
  {
    float t=euclideanRhythmsInteg(6.,15.,time.y/(.25 beat))*(.25 beat);
    dest+=.3*mix(.5,1.,sidechain)*pan(.3)*exp(-t*200.)*sin(17000.*t);
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

  // crash
  {
    float t=time.z;

    float env=mix(exp(-1.*t),exp(-10.*t),.8);

    dest+=.3*mix(.3,1.,sidechain)*tanh(10.*shotgun(3800.*t,1.9,.2)*env);
  }

  // bass
  {
    float st=mod(floor(time.y/(.25 beat)),8.);
    float t=euclideanRhythmsInteg(5.,8.,time.y/(.25 beat)-6.)*(.25 beat);
    float rest=euclideanRhythmsRest(5.,8.,time.y/(.25 beat)-6.)*(.25 beat);

    float env=linearstep(0.,.001,t)*linearstep(0.,.01,rest-.05 beat);
    float phase=p2f(24.+TRANSPOSE)*t;
    vec2 uv=orbit(phase);
    float wave=mix(
      tanh(10.*sin(TAU*phase)),
      tanh(4.-8.*texture(image_fbm,uv).x),
      0.2
    );
    dest+=.4*sidechain*env*wave;
  }

  return tanh(1.5*dest);
}
