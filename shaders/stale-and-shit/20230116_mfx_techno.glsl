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
const float SQRT2=sqrt(2.);
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

vec2 cheapnoise(float t){
  const float PERIOD=1./256.;
  float p=fract(t/PERIOD);

  vec3 dice;
  vec2 v=vec2(0);

  dice=pcg3df(vec3(lofi(t,PERIOD)))-vec3(.5,.5,0);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));
  dice=pcg3df(vec3(lofi(t+PERIOD,PERIOD)))-vec3(.5,.5,1);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));
  dice=pcg3df(vec3(lofi(t+PERIOD+PERIOD,PERIOD)))-vec3(.5,.5,2);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));

  return 2.*v;
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

      dest+=.6*env*tanh(2.*sin(
        300.*t
        -20.*exp(-40.*t)
        -5.*exp(-400.*t)
      ));
    }
  }

  // -- sub kick -----------------------------------------------------------------------------------
  {
    float t=mod(time.x-.25*b2t,.25*b2t);

    float zc=linearstep(0.,.002,t)*linearstep(0.,.002,.25*b2t-t);
    float env=exp(-10.*t);
    float wave=sin(310.*t-2.*exp(-80.*t));
    dest+=.5*sidechain*zc*env*wave;
  }

  // -- low freq noise -----------------------------------------------------------------------------
  {
    float t=time.x;
    vec2 uv=orbit(8.*t)+t*1.3;

    vec2 wave=vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.01).x
    )-.5;
    dest+=.5*sidechain*wave;
  }

  // -- hihat --------------------------------------------------------------------------------------
  {
    float t=mod(time.x-.5*b2t,1.*b2t);
    float decay=20.;
    dest+=.2*tanh(8.*shotgun(5400.*t,1.4,.0))*exp(-decay*t);
  }

  // -- ride ---------------------------------------------------------------------------------------
  {
    float t=mod(time.y,.5*b2t);

    dest+=.1*sidechain*tanh(10.*shotgun(3200.*t,3.4,.1))*exp(-10.*t);
  }

  // -- perc ---------------------------------------------------------------------------------------
  {
    float tp=mod(time.y,2.*b2t);
    float t=mod(mod(tp,.75*b2t),.5*b2t);
    float st=tp-t;
    vec3 dice=pcg3df(vec3(st+6.));

    vec2 uv=orbit(exp2(6.+2.*dice.x)*t)+exp2(5.+3.*dice.y)*t;
    vec2 wave=vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5;

    float env=mix(
      exp(-30.*t),
      exp(-5.*t),
      0.2
    );
    dest+=.2*sidechain*env*tanh(10.*wave);
  }

  // -- perc 2 -------------------------------------------------------------------------------------
  {
    float t=mod(time.x,.25*b2t);
    float st=floor(time.y/.25/b2t);
    vec2 wave=cheapnoise(40.*t)-cheapnoise(40.*t+.1*exp2(-3.+10.*t));
    //dest+=.3*exp(-10.*t)*sidechain*wave;
  }

  // -- clav ---------------------------------------------------------------------------------------
  {
    float t=mod(mod(time.y,2.25*b2t),.5*b2t);
    dest+=.2*r2d(1.4)*exp(-t*200.)*vec2(sin(17000.*t));
  }

  // -- rim ----------------------------------------------------------------------------------------
  {
    float t=mod(mod(time.y-.25*b2t,1.25*b2t),.5*b2t);

    float env=exp(-300.*t);

    dest+=.3*r2d(-1.4)*env*tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    ));
  }

  // -- noise --------------------------------------------------------------------------------------
  {
    float t=mod(time.z,32.*b2t);
    float tt=500.0*exp(-.2*t);
    float tt2=500.0*exp(-.2*(t+.00005*(1.-exp(-.2*t))));
    vec2 wave=cheapnoise(tt)-cheapnoise(tt2);
    dest+=.1*sidechain*wave;
  }

  // -- signal -------------------------------------------------------------------------------------
  {
    vec2 sum=vec2(0);

    for(int i=0;i<8;i++){
      float fi=float(i);

      const float freqs[3]=float[](560.,1200.,240.);
      const float times[3]=float[](.25,.75,1.5);

      for(int j=0;j<3;j++){
        float t=mod(time.z-times[j]*b2t-.5*b2t*fi,2.*b2t);
        vec2 wave=vec2(sin(TAU*freqs[j]*t));
        wave+=1.*vec2(sin(2.*TAU*freqs[j]*t+1.*wave)); // osc 2
        sum+=exp(-30.*max(t-.05,0.))*exp(-2.*fi)*wave*r2d(2.1*t2b*time.z-fi);
      }
    }

    dest+=.1*sum;
  }

  return tanh(1.5*dest);
}
