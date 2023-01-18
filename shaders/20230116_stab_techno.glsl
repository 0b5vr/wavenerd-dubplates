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

const float TRANSPOSE=-2.;

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

    if(inrange(time.z,0.,61.*b2t)){
      float env=linearstep(0.3,0.1,t);
      // env*=exp(-100.*t); // hi pass like

      dest+=.5*env*tanh(2.*sin(
        310.*t-55.*exp(-30.*t)
        -30.*exp(-500.*t)
      ));
    }
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t=mod(time.x,.25*b2t);
    float st=mod(floor(time.y/(.25*b2t)),16.);
    float decay=exp2(7.-3.*fract(st*.63));
    dest+=.2*tanh(8.*shotgun(5000.*t,2.,.0))*exp(-decay*t);
  }

  // -- clap ---------------------------------------------------------------------------------------
  {
    float t=mod(time.y-b2t,2.*b2t);

    float env=mix(
      exp(-30.*t),
      exp(-200.*mod(t,.013)),
      exp(-80.*max(0.,t-.02))
    );

    vec2 uv=orbit(46.*t)+40.*t;

    dest+=.23*tanh(20.*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5));
  }

  // -- ride ---------------------------------------------------------------------------------------
  {
    float t=mod(time.y,.5*b2t);

    dest+=.1*sidechain*tanh(10.*shotgun(3000.*t,3.4,.2))*exp(-10.*t);
  }

  // -- rim ----------------------------------------------------------------------------------------
  {
    float t=mod(mod(time.y-.25*b2t,1.25*b2t),.5*b2t);

    float env=exp(-300.*t);

    dest+=.3*env*tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    ))*vec2(1,-1);
  }

  // -- crash --------------------------------------------------------------------------------------
  {
    float t=time.z;
    dest+=.2*mix(.2,1.,sidechain)*tanh(8.*shotgun(4000.*t,3.,.0))*mix(exp(-t),exp(-10.*t),.5);
  }

  // -- bass ---------------------------------------------------------------------------------------
  {
    float t=mod(time.x,.25*b2t);
    float ptn[8]=float[](
      0.,7.,0.,5.,
      0.,10.,12.,0.
    );
    int st=int(time.z*4.*t2b)%8;

    // sub
    {
      float freq=p2f(24.+TRANSPOSE+ptn[st]);
      dest+=.2*sidechain*zcross(t,.25*b2t,1E-3)*sin(2.*sin(TAU*t*freq));
    }

    // unison square
    for(int i=0;i<16;i++){
      vec3 dice=pcg3df(vec3(i,st,0));
      vec2 dicen=boxMuller(dice.xy);

      float freq=p2f(36.+TRANSPOSE+ptn[st]+.1*dicen.x);
      float phase=TAU*t*freq+dice.y;

      float fm=2.*exp(-30.*t)*sin(2.*phase);
      float car=exp(-5.*t)*sin(phase+fm);
      dest-=.08*sidechain*zcross(t,.25*b2t,1E-3)*vec2(tanh(4.*car))*r2d(TAU*dice.z);
    }
  }

  // -- chord --------------------------------------------------------------------------------------
  {
    float chord[4]=float[](
      0.,2.,3.,7.
    );
    vec2 sum=vec2(0);

    float t=mod(time.y-.25*b2t,2.*b2t);

    for(int i=0;i<64;i++){
      float fi=float(i);
      vec3 dice=pcg3df(vec3(i));
      vec2 dicen=boxMuller(dice.xy);

      float freq=p2f(60.+TRANSPOSE+chord[i%4]+.1*dicen.x);
      float phase=freq*t+dice.x;
      vec2 uv=.1*exp(-5.*t)*orbit(phase)+.01*time.w;
      float wave=sin(TAU*phase);
      wave+=.6*sin(3.*TAU*phase);
      wave+=.3*sin(4.*TAU*phase);
      wave+=.02*sin(11.*TAU*phase);
      float env=mix(
        exp(-20.*max(t-.04,0.)),
        exp(-5.*t),
        .2
      );
      sum+=vec2(wave)*r2d(fi)*env;
    }

    dest+=.1*mix(.2,1.,sidechain)*sum;
  }

  return tanh(1.5*dest);
}
