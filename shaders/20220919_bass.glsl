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

uniform sampler2D image_fbm;

float TRANSPOSE=1.;

const float PI=acos(-1.);
const float TAU=2.*PI;
const float LN2=log(2.);

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform vec4 param_knob2;
uniform vec4 param_knob3;

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

vec2 orbit(float t){
  return vec2(cos(TAU*t),sin(TAU*t));
}

bool euclideanRhythms(float pulses,float steps,float i){
  float t=mod(i*pulses,steps);
  return t-pulses<0.;
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

vec2 mainAudio(vec4 time){
  vec2 dest=vec2(0);

  float kickt;
  float sidechain;
  
  TRANSPOSE+=.2*sin(time.z*3.);

  // kick
  {
    float t=kickt=time.x;
    t=kickt=time.y<2. beat?time.x:mod(time.y-2. beat,.75 beat);
    // t=euclideanRhythmsInteg(7.,16.,time.z*4.*TIME2BEAT-2.)/(4.*TIME2BEAT);
    sidechain=1.-smoothstep(1. beat,.0,t)*smoothstep(.0,.001,time.x);

    if(inrange(time.z,0. beat,61. beat)){
      float env=linearstep(.0,.001,t)*exp(-1.*t);

      dest+=.5*tanh((
        env*tanh(1.5*sin(TAU*p2f(TRANSPOSE+24.)*t-45.*exp(-35.*t)))
        +tanh(1.5*sin(TAU*3.*exp(-500.*t)))
      ));
    }
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);
    t=kickt<.25 beat?mod(t,.0625 beat):t;
    float st=mod(floor(time.z/(.25 beat)),16.);

    float env=exp(-50.*t);

    dest+=.2*mix(.5,1.,sidechain)*tanh(10.*shotgun(2700.*t,2.5,.3)*env);
  }

  // clap
  {
    float t=kickt-.5 beat;
    t=kickt;
    t=mod(time.y-1. beat,2. beat);
    t=mod(time.y-2. beat,4. beat);

    if(t>0.){
      float env=mix(
        exp(-30.*t),
        exp(-200.*mod(t,.013)),
        exp(-80.*max(0.,t-.02))
      );

      vec2 uv=orbit(127.*t)+20.*t;

      dest+=.23*tanh(20.*env*(vec2(
        texture(image_fbm,uv).x,
        texture(image_fbm,uv+.05).x
      )-.5));
    }
  }

  // rim
  {
    float st=floor(time.y/(.25 beat));
    float t=euclideanRhythmsInteg(10.,16.,time.y*4.*TIME2BEAT-2.)*.25 beat;
    t=fract(st*.643+.8)<.2?mod(t,.125 beat):t;

    float env=exp(-300.*t);

    dest+=.4*env*mix(.5,1.,sidechain)*tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    ));
  }

  // fm perc
  {
    float t=lofi(mod(time.x,.25 beat),1E-4);
    vec3 dice=pcg3df(vec3(lofi(time.z,.25 beat)));
    float freq=mix(200.,1000.,dice.x*dice.x*dice.x*dice.x);
    float decay=mix(50.,10.,dice.y*dice.y);
    float fmamp=mix(10.,100.,dice.z*dice.z);
    float a=sin(fmamp*sin(freq*exp(-t))*exp(-decay*t));
    dest+=.1*mix(.5,1.,sidechain)*a;
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

  // crash
  {
    float t=time.z;

    float env=mix(exp(-1.*t),exp(-10.*t),.8);

    dest+=.3*mix(.3,1.,sidechain)*tanh(10.*shotgun(3950.*t,1.9,.2)*env);
  }

  // rave hit
  {
    vec2 sum = vec2(0.);

    int pitchTable[8]=int[](0,3,7,10,12,14,19,22);
    float t=euclideanRhythmsInteg(7.,16.,time.z*4.*TIME2BEAT-2.)/(4.*TIME2BEAT);
    float l=t+euclideanRhythmsRest(7.,16.,time.z*4.*TIME2BEAT-2.)/(4.*TIME2BEAT);

    float env=exp(-3.*t)*smoothstep(0.,.001,t)*smoothstep(0.,.001,l-t);

    float noteb=mod(time.z*4.*TIME2BEAT,32.);
    int note=noteb<5.?1:0;

    for(int i=0;i<48;i++){
      float fi=float(i);
      vec3 dice=pcg3df(vec3(i));

      float freq=p2f(TRANSPOSE+float(note+48+pitchTable[i%8]))*mix(0.99,1.01,dice.x);
      float offu=dice.y;

      vec2 uv=vec2(1.*t+.01*fi);
      vec2 uv1=uv+.05*orbit(freq*t);
      vec2 uv2=uv+.05*orbit(freq*t+.4);
      vec2 uvm=.05*exp(-5.*t)*orbit(3.*freq*t);
      float diff=texture(image_fbm,uv1+uvm).x-texture(image_fbm,uv2+uvm).x;

      sum+=env*vec2(diff)*r2d(TAU*dice.z); // fbm osc
    }

    dest+=.3*mix(.5,1.,sidechain)*tanh(.5*sum);
  }

  return tanh(2.*dest);
}
