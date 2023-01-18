#define b2t (60./bpm)
#define t2b (1./b2t)

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define inrange(x,a,b) ((a)<=(x)&&(x)<(b))

const float TRANSPOSE=3.;

const float PI=acos(-1.);
const float TAU=2.*PI;
const float P4=pow(2.,5./12.);
const float P5=pow(2.,7./12.);

uniform sampler2D image_fbm;

float zcross( float t, float l, float d ) {
  return linearstep( 0.0, d, t ) * linearstep( 0.0, d, l - t );
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

vec2 orbit(float t){
  return vec2(cos(TAU*t),sin(TAU*t));
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

float cheapfiltersaw(float phase,float k){
  float wave=mod(phase,1.);
  float c=.5+.5*cos(PI*saturate(wave/k));
  return (wave+c)*2.-1.-k;
}

vec2 boxMuller(vec2 xi){
  float r=sqrt(-2.*log(xi.x));
  float t=xi.y;
  return r*orbit(t);
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2(0);

  // -- kick ---------------------------------------------------------------------------------------
  float kickt;
  float sidechain;

  {
    float t=kickt=time.x;
    sidechain=smoothstep(0.,.8*b2t,t);

    {
      float env=linearstep(0.3,0.1,t);
      // env*=exp(-50.*t); // hi pass like

      dest+=.5*env*tanh(3.*sin(
        310.*t-20.*exp(-28.*t)
        -20.*exp(-500.*t)
      ));
    }
  }

  // bass
  {
    float t=time.x;
    vec2 uv=orbit(8.*t)+t*1.3;

    vec2 wave=vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.01).x
    )-.5;
    wave+=sin(320.*t);
    dest+=.4*sidechain*wave;
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t=mod(time.x-.5*b2t,.25*b2t);
    float st=floor(time.y*4.*t2b);
    float decay=exp2(7.-3.*fract(.628*st));
    dest+=.2*tanh(8.*shotgun(5400.*t,1.4,.0))*exp(-decay*t);
  }

  // -- clap ---------------------------------------------------------------------------------------
  {
    float t=mod(time.y-b2t,2.*b2t);
    t=lofi(t,1E-4);

    float env=mix(
      exp(-30.*t),
      exp(-200.*mod(t,.013)),
      exp(-80.*max(0.,t-.02))
    );

    vec2 uv=orbit(87.*t)+20.*t;

    dest+=.23*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }

  // -- ride ---------------------------------------------------------------------------------------
  {
    float t=mod(time.y,.5*b2t);

    float env=mix(
      exp(-5.*t),
      exp(-50.*t),
      .2
    );
    dest+=.1*mix(.3,1.,sidechain)*tanh(10.*shotgun(3000.*t,3.4,.3))*env;
  }

  // -- crash --------------------------------------------------------------------------------------
  {
    float t=time.z;
    dest+=.2*mix(.2,1.,sidechain)*tanh(8.*shotgun(4000.*t,3.,.0))*mix(exp(-t),exp(-10.*t),.5);
  }

  // -- ep -----------------------------------------------------------------------------------------
  {
    vec2 sum=vec2(0);

    for(int i=0;i<28;i++){
      float delay=float(i/7);

      float chord[7]=float[](0.,2.,3.,7.,9.,10.,14.);
      float note=chord[i%7];

      float tp=mod(time.z-.25*b2t-.5*b2t*delay,16.*b2t);
      float t=mod(mod(tp,15.5*b2t),.75*b2t);
      int st=int((tp-t)*t2b*4.+.1);

      vec3 dice=pcg3df(vec3(i,st,0));
      vec2 dicen=boxMuller(dice.xy);

      const float progs[4]=float[](0.,2.,5.,4.);
      float prog=progs[st/16];

      float freq=p2f(48.+prog+TRANSPOSE+note+.02*dicen.y);
      float phase=lofi(freq*t+TAU*dice.z,1./32.);
      vec2 wave=vec2(
        +.25*sin(TAU*phase)
        +.25*sin(2.*TAU*phase)
        +.14*sin(3.*TAU*phase)
        +.10*sin(4.*TAU*phase)
      )*vec2(1,-1);

      float env=exp(-50.*max(t-.15,0.));
      float zc=linearstep(0.,1E-3,t)*linearstep(0.,1E-3,.75*b2t-t);

      sum+=zc*env*exp(-2.*delay)*wave*r2d(-3.*delay+0.3*dicen.x);
    }

    dest+=.2*sum;
  }

  return tanh(1.5*dest);
}
