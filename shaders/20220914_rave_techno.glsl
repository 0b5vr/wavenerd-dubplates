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

const float TRANSPOSE=-3.;

const float PI=acos(-1.);
const float TAU=2.*PI;
const float LN2=log(2.);

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform vec4 param_knob2;
uniform vec4 param_knob3;

vec2 orbit( float t ) {
  return vec2( cos( TAU * t ), sin( TAU * t ) );
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

bool euclideanRhythms( float pulses, float steps, float i ) {
  float t = mod( i * pulses, steps );
  return t - pulses < 0.0;
}

float euclideanRhythmsInteg( float pulses, float steps, float time ) {
  float t = mod( floor( time ) * pulses, steps );
  return floor( ( t - pulses ) / pulses ) + 1.0 + fract( time );
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

  float sidechain;

  // kick
  {
    float t=time.x;
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

    float envptn=fract(.4+.63*st);
    float env=exp(-exp2(4.5+2.*envptn)*t);

    dest+=.2*sidechain*tanh(10.*shotgun(5000.0*t,2.5,.3)*env);
  }

  // hihat
  {
    float t=mod(time.x-.5 beat,1. beat);

    float env=exp(-10.*t);

    dest+=.2*sidechain*tanh(10.*shotgun(3000.0*t,2.6,.0)*env);
  }

  // clap
  {
    float t=mod(time.y-1. beat,2. beat);

    float env=mix(
      exp(-10.*t),
      exp(-200.*mod(t,.013)),
      exp(-100.*max(0.,t-.02))
    );

    vec2 uv=orbit(89.*t)+40.*t;

    dest+=.23*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }

  // rim
  {
    float t=euclideanRhythmsInteg(10.,16.,time.y*4.*TIME2BEAT-2.)*.25 beat;

    float env=exp(-300.*t);

    dest+=.4*env*tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    ));
  }

  // fm perc
  {
    float t=lofi(mod(time.x,.25 beat),1E-4);
    float st=floor(time.z/(.25 beat));
    vec3 dice=pcg3df(vec3(st));

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

  // bass
  {
    float t=euclideanRhythmsInteg(5.,16.,time.z*4.*TIME2BEAT-2.)/(4.*TIME2BEAT);

    float noteb=mod(time.z*4.*TIME2BEAT,32.);
    int note=noteb<9.?0:
      noteb<16.?3:
      noteb<25.?-2:
      1;

    float freq=p2f(TRANSPOSE+float(note+36));

    dest+=.4*sidechain*tanh(9.*exp(-10.*t)*sin(TAU*freq*t))*smoothstep(0.,.01,t);
  }

  // crash
  {
    float t=time.z;

    float env=mix(exp(-1.*t),exp(-10.*t),.8);

    dest+=.3*mix(.3,1.,sidechain)*tanh(10.*shotgun(3600.*t,1.9,.2)*env);
  }

  // rave hit
  {
    vec2 sum = vec2(0.);

    int pitchTable[8]=int[](0,3,7,10,12,14,19,22);
    float t=euclideanRhythmsInteg(7.,16.,time.z*4.*TIME2BEAT-2.)/(4.*TIME2BEAT);

    float noteb=mod(time.z*4.*TIME2BEAT,32.);
    int note=noteb<9.?0:
      noteb<16.?3:
      noteb<25.?-2:
      1;

    for(int i=0;i<96;i++){
      float fi=float(i);
      vec3 dice=pcg3df(vec3(i));

      float freq=p2f(TRANSPOSE+float(note+48+pitchTable[i%8]))*mix(0.99,1.01,dice.x);
      float offu=dice.y;

      vec2 uv=vec2(1.*t+.1*fi);
      vec2 uv1=uv+0.1*orbit(freq*t);
      vec2 uv2=uv+0.05*orbit(3.*freq*t);
      vec2 uvm=.1*exp(-10.*t)*orbit(9.*freq*t);
      float diff=texture(image_fbm,uv1+uvm).x-texture(image_fbm,uv2+uvm).x;

      sum+=vec2(diff)*r2d(PI*(.5+dice.z)); // fbm osc
    }

    dest+=.37*mix(.5,1.,sidechain)*tanh(.5*exp(-3.*t)*sum);
  }

  return tanh(2.*dest);
}
