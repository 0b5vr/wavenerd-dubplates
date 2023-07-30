#define b2t (60./bpm)
#define t2b (1./b2t)

#define saturate(i) clamp(i,0.,1.)
#define clip(i) clamp(i,-1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define repeat(i,n) for(int i=0;i<(n);i++)

const float pi=acos(-1.);
const float tau=2.*pi;
const float p4=exp2(5./12.);
const float p5=exp2(7./12.);
const uint uint_max=0xffffffffu;

uvec3 pcg3d(uvec3 v){
  v=v*1145141919u+1919810u;
  v.x+=v.y*v.z;
  v.y+=v.z*v.x;
  v.z+=v.x*v.y;
  v^=v>>16u;
  v.x+=v.y*v.z;
  v.y+=v.z*v.x;
  v.z+=v.x*v.y;
  return v;
}

vec3 pcg3df(vec3 v){
  uvec3 r=pcg3d(floatBitsToUint(v));
  return vec3(r)/float(uint_max);
}

mat2 r2d(float x){
  float c=cos(x),s=sin(x);
  return mat2(c,s,-s,c);
}

mat3 orthbas(vec3 z){
  z=normalize(z);
  vec3 x=normalize(cross(vec3(0,1,0),z));
  vec3 y=cross(z,x);
  return mat3(x,y,z);
}

vec3 cyclic(vec3 p,float pump){
  vec4 sum=vec4(0);
  mat3 rot=orthbas(vec3(2,-3,1));

  repeat(i,5){
    p*=rot;
    p+=sin(p.zxy);
    sum+=vec4(cross(cos(p),sin(p.yzx)),1);
    sum*=pump;
    p*=2.;
  }

  return sum.xyz/sum.w;
}

vec2 orbit(float t){
  return vec2(cos(tau*t),sin(tau*t));
}

vec2 shotgun(float t,float spread,float snap){
  vec2 sum=vec2(0);
  repeat(i,64){
    vec3 dice=pcg3df(vec3(i));

    float partial=exp2(spread*dice.x);
    partial=mix(partial,floor(partial+.5),snap);

    sum+=vec2(sin(tau*t*partial))*r2d(tau*dice.y);
  }
  return sum/64.;
}

float cheapfiltersaw(float phase,float k){
  float wave=mod(phase,1.);
  float c=.5+.5*cos(pi*saturate(wave/k));
  return (wave+c)*2.-1.-k;
}

vec2 boxmuller(vec2 xi){
  float r=sqrt(-2.*log(xi.x));
  float t=xi.y;
  return r*orbit(t);
}

vec2 cheapnoise(float t){
  uvec3 s=uvec3(t*256.);
  float p=fract(t*256.);

  vec3 dice;
  vec2 v=vec2(0);

  dice=vec3(pcg3d(s))/float(uint_max)-vec3(.5,.5,0);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));
  dice=vec3(pcg3d(s+1u))/float(uint_max)-vec3(.5,.5,1);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));
  dice=vec3(pcg3d(s+2u))/float(uint_max)-vec3(.5,.5,2);
  v+=dice.xy*smoothstep(1.,0.,abs(p+dice.z));

  return 2.*v;
}

vec2 mainAudio(vec4 time){
  vec2 dest=vec2(0);

  float sidechain=1.;

  { // kick
    float t=time.x;
    sidechain=smoothstep(0.,1E-3,b2t-t)*smoothstep(0.,.8*b2t,t);

    if(time.z<61.*b2t){
      float env=linearstep(0.3,0.1,t);

      // { // hi pass like
      //   env*=exp(-100.*t);
      // }

      dest+=.6*env*tanh(2.*sin(
        310.*t-55.*exp(-30.*t)
        -30.*exp(-500.*t)
      ));
    }
  }

  { // bass
    float t=mod(time.x,.25*b2t);
    float ptn[7]=float[](
      0.,12.,0.,17.,
      0.,13.,12.
    );
    int st=int(time.z*4.*t2b)%7;

    vec2 sum=vec2(0);

    { // sub
      float freq=p2f(22.+ptn[st]);
      sum+=.3*sin(2.*sin(tau*t*freq));
    }

    repeat(i,16){ // unison fm
      vec3 dice=pcg3df(vec3(i,st,0));
      float freq=p2f(37.+ptn[st]+.2*(dice.x-.5));
      float phase=tau*t*freq+dice.y;

      vec2 fm2=.2*exp(-10.*t)*vec2(sin(7.77*phase))*r2d(tau*dice.x);
      vec2 fm=8.*exp(-3.*t)*vec2(sin(.5*p5*phase+fm2))*r2d(tau*dice.y);
      vec2 car=exp(-8.*t)*vec2(sin(phase+fm))*r2d(tau*dice.z);
      sum+=.1*car;
    }

    float zc=linearstep(0.,1E-3,t)*linearstep(0.,1E-3,.25*b2t-t);
    dest+=1.*sidechain*zc*sum;
  }

  { // hihat
    float t=mod(time.x,.25*b2t);
    float decay=time.y<3.75*b2t?90.:10.;
    float env=exp(-decay*t);
    dest+=.25*env*tanh(8.*shotgun(4000.*t,2.,.2));
  }

  { // clap
    float t=mod(time.y-b2t,2.*b2t);

    float env=mix(
      exp(-30.*t),
      exp(-200.*mod(t,.013)),
      exp(-80.*max(0.,t-.02))
    );

    vec2 wave=cyclic(vec3(4.*orbit(200.*t),440.*t),1.5).xy;
    dest+=.2*tanh(20.*env*wave);
  }

  { // ride
    float t=mod(time.y,.5*b2t);

    dest+=.15*sidechain*tanh(10.*shotgun(4200.*t,2.4,.4))*exp(-10.*t);
  }

  { // psysaw
    float t=mod(time.y,.25*b2t);
    int st=int(time.z*4.*t2b);
    vec3 dice=pcg3df(vec3(st));
    float l=(.25-dice.y*.2)*b2t;
    float freq=20.*sin(tau*dice.z*2.);
    float fm=fract(10.*exp(-freq*t));
    float wave=fract(20.*exp(-2.*fm))-.5;
    float zc=linearstep(0.,1E-3,t)*linearstep(0.,1E-3,l-t);
    dest+=.2*sidechain*zc*wave;
  }

  { // crash
    float t=time.z;
    float env=mix(exp(-t),exp(-10.*t),.5);
    vec2 wave=shotgun(4000.*t,3.,.0);
    dest+=.3*mix(.2,1.,sidechain)*tanh(8.*wave)*env;
  }

  // { // chord
  //   float chord[8]=float[](
  //     0.,5.,7.,12.,14.,19.,22.,29.
  //   );
  //   vec2 sum=vec2(0);

  //   float t=time.z;

  //   repeat(i,64){
  //     vec3 dice=pcg3df(vec3(i));

  //     float freq=p2f(57.+chord[i%8]+.1*boxmuller(dice.xy).x);
  //     float phase=freq*t;
  //     float wave=cheapfiltersaw(phase,.02)-cheapfiltersaw(phase,.2);
  //     sum+=vec2(wave)*r2d(tau*dice.z);
  //   }

  //   dest+=.9*mix(.2,1.,sidechain)*sum/32.;
  // }

  return clip(1.3*tanh(dest));
}
