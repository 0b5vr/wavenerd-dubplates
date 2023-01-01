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

mat2 r2d(float x){
  float c=cos(x),s=sin(x);
  return mat2(c, s, -s, c);
}

float euclideanRhythmsInteg(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((t-pulses)/pulses)+1.+fract(time);
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
    float t=mod(time.z,8. beat);
    float haha=mod(mod(time.y-2. beat,4. beat),.75 beat);
    t=kickt=(
      t<2. beat?time.x:
      t<5. beat?haha:
      t<6. beat?time.x:
      haha
  	);
    sidechain=1.-smoothstep(1. beat,.0,t)*linearstep(.0,.001,t);

    float env=linearstep(.0,.001,t)*linearstep(0.3,0.1,t);

    dest+=.5*env*tanh(1.5*sin(300.*t-100.*exp(-30.*t)-100.*exp(-200.*t)));
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),16.);

    float envptn=fract(.4+.63*st);
    float env=exp(-exp2(4.5+2.*envptn)*t);

    dest+=.2*sidechain*tanh(10.*shotgun(6000.*t,1.5,.2)*env);
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

  // crash
  {
    float t=time.z;

    float env=mix(exp(-1.*t),exp(-10.*t),.8);

    dest+=.3*mix(.3,1.,sidechain)*tanh(10.*shotgun(3800.*t,1.9,.2)*env);
  }

  // bass
  {
    float freq=p2f(24.+TRANSPOSE);

    dest+=.3*sidechain*tanh(4.*sin(TAU*freq*kickt));
  }

  // pad
  {
    vec2 sum = vec2(0);

    int pitchTable[7]=int[](0,3,7,10,14,17,21);

    for (int i=0;i<48;i++){
      float fi=float(i);
      vec3 dice=pcg3df(vec3(i));

      float t=time.z;

      float freq=p2f(48.+TRANSPOSE+float(pitchTable[i%7]))*mix(.99,1.01,fs(fi));
      float offu=fs(fi+4.);

      vec2 uv=vec2(.5);
      uv+=.2*orbit(freq*t+offu);
      uv+=sidechain*.04*orbit(9.*freq*t+offu);
      uv+=sidechain*.02*orbit(14.*freq*t+offu);
      float wave=mix(-1.,1.,texture(image_fbm,uv).x);
      wave-=mix(-1.,1.,texture(image_fbm,uv+.2).x);

      float amp=.1*mix(.0,1.,sidechain);
      sum+=amp*vec2(wave)*r2d(TAU*dice.z); // fbm osc
    }

    dest+=clip(sum);
  }

  // arp
  {
    vec2 sum = vec2(0);

    int pitchTable[5]=int[](0,7,5,14,10);

    for (int i=0;i<4;i++){
      float fi=float(i);

      float t=mod(time.x,.25 beat);
      float st=mod(floor(time.z/(.25 beat)-fi*3.),32.);

      float arpptn1=mod(st*2.1,14.);
      float pitch1=48.+TRANSPOSE+float(pitchTable[int(arpptn1)%5]);
      pitch1+=floor(arpptn1/5.)*12.;

      float phase=t*p2f(pitch1);
      vec2 wave=saw(phase+vec2(.0,.8));
      wave-=saw(phase*1.501+2.+vec2(.0,.8));
      float amp=.2*mix(.3,1.,sidechain)*linearstep(.0,.001,t)*exp(-10.*t)*exp(-fi);
      sum+=amp*tanh(5.*wave);
    }

    dest+=mix(.3,1.,sidechain)*clip(sum);
  }

  return tanh(1.5*dest);
}
