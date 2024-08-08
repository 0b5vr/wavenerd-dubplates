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

const float TRANSPOSE=-3.;

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
    float decay=time.y<3.75*b2t?90.:10.;
    dest+=.2*tanh(8.*shotgun(4000.*t,2.,.2))*exp(-decay*t);
  }

  // -- clap ---------------------------------------------------------------------------------------
  {
    float t=mod(time.y-b2t,2.*b2t);

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

    dest+=.15*sidechain*tanh(10.*shotgun(4200.*t,2.4,.4))*exp(-10.*t);
  }

  // -- psysaw -------------------------------------------------------------------------------------
  {
    float t=mod(time.y,.25*b2t);
    int st=int(time.z*4.*t2b);
    vec3 dice=pcg3df(vec3(st));
    float l=(.25-dice.y*.2)*b2t;
    float freq=20.*sin(TAU*dice.z*2.);
    dest+=sidechain*.1*zcross(t,l,1E-3)*saw(20.*exp(-2.*fract(10.*exp(-freq*t))));
  }

  // -- crash --------------------------------------------------------------------------------------
  {
    float t=time.z;
    dest+=.2*mix(.2,1.,sidechain)*tanh(8.*shotgun(4000.*t,3.,.0))*mix(exp(-t),exp(-10.*t),.5);
  }

  // -- bass ---------------------------------------------------------------------------------------
  {
    float t=mod(time.x,.25*b2t);
    float ptn[7]=float[](
      0.,12.,0.,17.,
      0.,13.,12.
    );
    int st=int(time.z*4.*t2b)%7;

    // sub
    {
      float freq=p2f(24.+TRANSPOSE+ptn[st]);
      dest+=.3*sidechain*zcross(t,.25*b2t,1E-3)*sin(2.*sin(TAU*t*freq));
    }

    // unison fm
    for(int i=0;i<16;i++){
      vec3 dice=pcg3df(vec3(i,st,0));
      float freq=p2f(39.+TRANSPOSE+ptn[st]+.2*(dice.x-.5));
      float phase=TAU*t*freq+dice.y;

      vec2 fm2=.2*exp(-10.*t)*vec2(sin(7.77*phase))*r2d(TAU*dice.x);
      vec2 fm=8.*exp(-3.*t)*vec2(sin(.5*P5*phase+fm2))*r2d(TAU*dice.y);
      vec2 car=exp(-8.*t)*vec2(sin(phase+fm))*r2d(TAU*dice.z);
      dest+=.14*sidechain*zcross(t,.25*b2t,1E-3)*car;
    }
  }

  // -- chord --------------------------------------------------------------------------------------
  {
    float chord[8]=float[](
      0.,5.,7.,12.,14.,19.,22.,29.
    );
    vec2 sum=vec2(0);

    float t=time.z;

    for(int i=0;i<64;i++){
      vec3 dice=pcg3df(vec3(i));

      float freq=p2f(60.+TRANSPOSE+chord[i%8]+.4*(dice.x-.5));
      float phase=freq*t;
      float wave=cheapfiltersaw(phase,.02)-cheapfiltersaw(phase,.2);
      sum+=vec2(wave)*r2d(TAU*dice.z);
    }

    // dest+=1.4*mix(.2,1.,sidechain)*sum/64.;
  }

  return tanh(1.5*dest);
}
