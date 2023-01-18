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
  return mat2(c,s,-s,c);
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
    dest+=.5*sidechain*(texture(image_fbm,uv).x-.5);
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
    float t=mod(mod(mod(time.y-1. beat,2. beat),.75 beat),.5 beat); // .xx. x.xx

    float env=exp(-300.*t);

    dest+=.2*env*tanh(4.*(
      +tri(t*400.-.5*env)
      +tri(t*1500.-.5*env)
    ))*vec2(1,-1);
  }

  // noise
  {
    float t=mod(time.x,.25 beat);
    float st=floor(time.z/(.25 beat));
    vec2 uv=exp(1.-4.*fs(st+.4))*orbit(exp(3.+3.*fs(st))*t);

    float env=exp(-100.0*fract(mod(st,32.)*.631+.5)*t);

    dest+=.2*sidechain*env*tanh(5.*(texture(image_fbm,uv).xx-.5))*r2d(st+.4);
  }

  // additive shepard
  {
    vec2 sum=vec2(0.);

    for(int i=0;i<2500;i++){
      vec3 diceA=pcg3df(vec3(i/50));
      vec3 diceB=pcg3df(vec3(i));

      float t=mod(time.z-diceA.x*(64. beat),64. beat);

      float tone=5.+8.*diceA.y+.15*diceB.y;
      float freq=exp2(tone);
      vec2 phase=(t+.5*t*t/(64. beat))*freq+fract(diceB.xy*999.);
      phase+=.1*fract(32.*phase); // add high freq

      sum+=sin(TAU*phase)*sin(PI*t/(64. beat))/1000.;
    }

    dest+=1.2*sidechain*sum;
  }

  return tanh(2.*dest);
}
