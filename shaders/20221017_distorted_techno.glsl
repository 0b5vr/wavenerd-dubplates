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
#define inrange(x,a,b) ((a)<=(x)&&(x)<(b))

const float TRANSPOSE=5.;

const float PI=acos(-1.);
const float TAU=2.*PI;
const float LN2=log(2.);

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform vec4 param_knob2;
uniform vec4 param_knob3;

uniform sampler2D image_fbm;

vec2 pan(float x){
  return mix(vec2(2,0),vec2(0,2),x);
}

mat2 r2d(float x){
  float c=cos(x),s=sin(x);
  return mat2(c,s,-s,c);
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

float euclideanRhythmsInteg(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((t-pulses)/pulses)+1.+fract(time);
}

float euclideanRhythmsRest(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((steps-t-1.)/pulses)+1.-fract(time);
}

vec2 shotgun(float t,float spread,float snap){
  vec2 sum=vec2(0);
  for(int i=0;i<64;i++){
    vec3 dice=pcg3df(vec3(i));

    float partial=exp2(spread*dice.x);
    partial=mix(partial,floor(partial+.5),snap);

    sum+=vec2(sin(TAU*t*partial))*r2d(TAU*dice.y);
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

  float kickt;

  // kick
  {
    float t=kickt=time.x;
    if(mod(time.z,8. beat)>7. beat){t=mod(t,.5 beat);}

    if(inrange(time.z,0. beat,61. beat)){
      float env=linearstep(0.5,0.1,t);

      dest+=.5*exp(-2.*t)*tanh(5.*(
        env*tanh(1.5*sin(
          TAU*p2f(24.+TRANSPOSE)*t-45.*exp(-35.*t)
        ))
      ));
    }
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),16.);

    float envptn=fract(.4+.63*st);
    float env=exp(-exp2(4.5+2.*envptn)*t);

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

  // crash
  {
    float t=time.z;

    float env=mix(exp(-1.*t),exp(-5.*t),.8);

    dest+=.2*tanh(10.*shotgun(3800.*t,1.9,.2)*env);
  }

  // chord
  {
    const int chord[3]=int[](0,3,7);

    vec2 sum=vec2(0);
    for(int iDelay=0;iDelay<3;iDelay++){
      float st=mod(time.y/(.25 beat)-2.1*float(iDelay),8.);
      float t=euclideanRhythmsInteg(3.,8.,st-3.)*(.25 beat);
      float rest=euclideanRhythmsRest(3.,8.,st-3.)*(.25 beat);
      float trans=mod(st-1.,8.)<2.?1.:0.;

      float filtenv=exp(-5.*t);
      float cutoff=p2f(120.*mix(filtenv,1.,.5)*exp(-.1*float(iDelay)));

      float env=linearstep(0.,.001,t)*linearstep(0.,.01,rest)*exp(-5.*t);
      vec2 p=pan(.5+.2*float(iDelay)*sin(float(iDelay)+time.w));
      vec2 amp=env*p*exp(-float(iDelay));

      for(int i=0;i<3;i++){
        float fi=float(i);
        float freq=p2f(48.+TRANSPOSE+trans+float(chord[i]))
          *(1.+.01*(fract(fi*.622)-.5));
        sum+=amp
          *(1.-2.*mod(fi,2.))
          *filterSaw(freq,t+vec2(0,3)+fi,cutoff,.1);
      }
    }

    dest+=.2*sum;
  }

  return tanh(1.5*dest);
}
