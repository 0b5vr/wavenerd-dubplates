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

const float TRANSPOSE=2.;

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

  float sidechain=mix(1.,.3,smoothstep(1. beat,.0,time.x)*linearstep(.0,.001,time.x));

  // kick
  if(inrange(time.z,0. beat,61. beat)){
    float t=time.x;

    float env=linearstep(.0,.001,t)*linearstep(0.3,0.1,t);

    dest+=.5*tanh((
      env*tanh(1.5*sin(360.*t-45.*exp(-35.*t)))
      +tanh(1.5*sin(TAU*3.*exp(-500.*t)))
    ));
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);

    float env=exp(-100.*t);
    env*=linearstep(.0,.001,t);

    vec2 uv=orbit(900.*t)+100.*t;

    dest+=.3*sidechain*tanh(5.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5));
  }

  // hihat
  {
    float t=mod(time.x-.5 beat,1. beat);

    float env=exp(-15.*t);

    vec2 uv=orbit(800.*t)+orbit(4000.*t)*exp(-100.*t)+137.*t;

    dest+=.2*sidechain*tanh(5.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5));

    uv=orbit(802.*t)+orbit(4000.*t)*exp(-100.*t)+137.*t;

    dest-=.2*sidechain*tanh(5.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5));
  }

  // clap
  {
    float t=mod(time.y-1. beat,2. beat);

    float env=mix(
      exp(-20.*t),
      exp(-200.*mod(t,.013)),
      exp(-100.*max(0.,t-.02))
    );

    vec2 uv=orbit(89.*t)+40.*t;

    dest+=.23*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }

  // tom
  {
    float t=mod(time.y-.25 beat,2. beat);

    float env=exp(-20.*t);
    env*=linearstep(.0,.001,t);

    dest+=vec2(.2,.1)*env*sin(vec2(0.,1.)+3.*sin(1100.*t-10.*exp(-100.*t)));
  }

  // tom
  {
    float t=mod(time.y-1.5 beat,2. beat);

    float env=exp(-20.*t);
    env *= linearstep(.0,.001,t);

    dest+=vec2(.1,.2)*env*sin(vec2(0.,5.)+3.*sin(800.*t-10.*exp(-100.*t)));
  }

  // rim
  {
    float t=euclideanRhythmsInteg(10.,16.,time.y*4.*TIME2BEAT-2.)*.25 beat;

    float env=exp(-300.*t);

    dest+=.3*env*tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    ))*mat2(0,1,-1,0);
  }

  // snare909
  if(inrange(time.z,60. beat,64. beat)){
    float t=mod(time.x,exp2(-2.-floor(time.y/(2. beat))) beat);

    vec2 uv=orbit(200.*t)+237.*t;
    vec2 wave=tanh(20.*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
    wave+=sin(1400.*t-exp(-t*80.)*30.);

    float amp=.14*smoothstep(-4. beat,4. beat,time.y);
    dest+=amp*tanh(2.*wave*exp(-20.*t));
  }

  // ride
  {
    float t=mod(time.x,.5 beat);

    float env=exp(-5.*t);

    dest+=.08*sidechain*tanh(10.*shotgun(4000.*t,2.5,.1)*env);
  }

  // crash
  {
    float t=time.z;

    float env=exp(-2.*t);

    vec2 uv=orbit(200.*t)+orbit(2000.*t)*exp(-20.*t)+1762.82*t;

    dest+=.2*sidechain*tanh(5.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.01).x
    )-.5));

    uv=orbit(201.*t)+orbit(2000.*t)*exp(-20.*t)+1762.82*t;

    dest-=.2*mix(.3,1.,sidechain)*tanh(5.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.01).x
    )-.5));
  }

  // acid
  {
    float t=mod(time.y,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),9.);

    float env=envAR(t,.25 beat,.001,.01);
    float cutoff=p2f(50.+50.*exp(-10.*t));
    vec2 wave=filterSaw(p2f(24.+TRANSPOSE),t,cutoff,.0);

    dest+=.2*env*sidechain*wave;
  }

  // acid
  {
    float t=mod(time.y,.25 beat);
    float st=mod(floor(time.z/(.25 beat)),9.);

    int arp[9]=int[](0,-12,-7,0,0,10,12,-7,5);
    float l=.25 beat*min(1.,mix(.5,1.2,fract(.62*st)));

    float freq=p2f(36.+TRANSPOSE+float(arp[int(st)]));
    float env=envAR(t,l,.001,.01);
    float cutoff=p2f(96.*mix(exp(-10.*t),1.,.7)+12.*sin(time.w));
    vec2 wave=filterSaw(freq,t,cutoff,.8);

    dest+=.2*env*sidechain*tanh(20.*wave);
  }

  // pad
  {
    vec2 sum = vec2(0);

    int pitchTable[7]=int[](0,3,7,10,12,14,17);

    for (int i=0;i<48;i++){
      float fi=float(i);
      vec3 dice=pcg3df(vec3(i));

      float t=mod(mod(mod(time.z,2. beat),.75 beat),.5 beat);

      float freq=p2f(48.+TRANSPOSE+float(pitchTable[i%7]))*mix(.99,1.01,dice.x);
      float offu=dice.y;

      vec2 uv=vec2(.3);
      uv+=.1*orbit(freq*t+offu);
      uv+=.08*exp(-5.*t)*orbit(5.*freq*t+offu);
      uv+=.02*exp(-5.*t)*orbit(17.*freq*t+offu);
      float wave=texture(image_fbm,uv).x;
      wave-=texture(image_fbm,uv+.07).x;

      float amp=.2*mix(.3,1.,sidechain)*exp(-5.*t);
      sum+=amp*vec2(wave)*r2d(.9*fi); // fbm osc
    }

    dest+=clip(sum);
  }

  // arp
  {
    vec2 sum = vec2(0);

    int pitchTable[5]=int[](0,7,5,10,17);

    for (int i=0;i<4;i++){
      float fi=float(i);

      float t=mod(time.x,.25 beat);
      float st=mod(floor(time.z/(.25 beat)-fi*3.),32.);

      float arpptn1=mod(st*6.5,17.);
      float pitch1=48.+TRANSPOSE+float(pitchTable[int(arpptn1)%5]);
      pitch1+=floor(arpptn1/5.)*12.;
      float arpptn0=mod((st-1.)*6.5,17.);
      float pitch0=48.+TRANSPOSE+float(pitchTable[int(arpptn0)%5]);
      pitch0+=floor(arpptn0/5.)*12.;

      float phase=glidephase(t,.03,pitch0,pitch1);
      vec2 wave=sin(TAU*phase+vec2(.0,.8));
      float amp=.1*mix(.3,1.,sidechain)*linearstep(.0,.001,t)*exp(-10.*t)*exp(-fi);
      sum+=amp*tanh(20.*wave*exp(-14.*t));
    }

    dest+=clip(sum);
  }

  return tanh(1.5*dest);
}
