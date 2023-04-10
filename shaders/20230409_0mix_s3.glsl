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
const float b2t=60./140.;
const float t2b=1./b2t;
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
    float tp=mod(time.y,2.*b2t);
    float t=mod(tp,.75*b2t);
    float l=mix(.75*b2t,.5*b2t,step(1.5*b2t,tp));
    sidechain=smoothstep(0.,1E-3,l-t)*smoothstep(0.,.8*b2t,t);

    float env=linearstep(.0,.001,t)*linearstep(0.3,0.1,t);

    float wave=mix(
      sin(300.*t-65.*exp(-80.*t)),
      sin(200.*t-15.*exp(-40.*t)),
      step(60.*b2t,time.z)
    );
    dest+=.6*tanh(3.*env*wave);
  }

  { // bass
    float t=mod(time.y,2.*b2t);

    vec2 sum=vec2(.5*sin(tau*45.*t));

    repeat(i,8){
      vec3 dice=pcg3df(vec3(i));
      float freq=45.+.1*boxmuller(dice.xy).x;
      float phase=freq*t+dice.z;
      float screech=2.*smoothstep(57.*b2t,61.*b2t,time.z);
      vec3 p=vec3(10.*t*orbit(phase),screech*sin(tau*31.*phase));
      sum+=.25*cyclic(p,4.).xy*r2d(tau*float(i)/8.+time.z);
    }

    dest+=.6*sidechain*tanh(sum);
  }

  { // hihat
    float t=mod(time.x,.25*b2t);
    float st=floor(time.y/.25/b2t);

    float env=exp(-40.*t);
    env*=linearstep(.0,.001,t);

    vec2 wave=cyclic(vec3(6000.*t),1.2).xy;

    dest+=.4*sidechain*env*tanh(5.*wave);
  }

  { // rim
    float t=mod(mod(mod(time.y-1.*b2t,2.*b2t),.75*b2t),.5*b2t); // .xx. x.xx

    float env=exp(-300.*t);
    dest+=.3*env*tanh(4.*(
      +tri(t*400.-.5*env)
      +tri(t*1500.-.5*env)
    ))*vec2(1,-1);
  }

  { // fm perc
    float t=mod(time.x,.25*b2t);
    t=lofi(t,1E-4);
    float st=floor(time.z/.25/b2t);
    vec3 dice=pcg3df(vec3(st));

    float freq=exp2(8.+3.*dice.x);
    float env=exp(-exp2(3.+5.*dice.y)*t);
    float fm=env*exp2(3.+3.*dice.z)*sin(freq*exp(-t));
    float wave=sin(fm);
    dest+=.1*mix(.2,1.,sidechain)*vec2(wave)*r2d(st);
  }

  { // hihat 2
    float t=mod(time.x,.25*b2t);
    float st=floor(time.y/.25/b2t);

    float env=exp(-exp2(3.+2.*fract(.4+.628*st))*t);
    env*=linearstep(.0,.001,t);

    vec2 wave=shotgun(4000.*t,3.,.5);

    dest+=.4*sidechain*env*tanh(5.*wave);
  }

  { // clap
    float t=mod(time.y-3.*b2t,4.*b2t);

    float env=exp(-40.*t)+.02*exp(-5.*t);

    t+=0.1*sin(t*90.0);
    vec3 p=vec3(10.*orbit(59.8*t),+250.*t);
    vec2 wave=cyclic(p,2.).xy;

    dest+=.2*tanh(20.*env*wave);
  }

  { // crash
    float t=time.z;
    float env=mix(exp(-t),exp(-10.*t),.5);
    vec2 wave=shotgun(4000.*t,3.,.0);
    dest+=.3*mix(.2,1.,sidechain)*tanh(8.*wave)*env;
  }

  { // additive shepard
    vec2 sum=vec2(0.);

    repeat(i,2500){
      vec3 diceA=pcg3df(vec3(i/50));
      vec3 diceB=pcg3df(vec3(i));

      float t=mod(time.z-diceA.x*(64.*b2t),64.*b2t);

      float tone=5.+8.*diceA.y+.15*diceB.y;
      float freq=exp2(tone);
      vec2 phase=(t+.5*t*t/(64.*b2t))*freq+fract(diceB.xy*999.);
      phase+=.1*fract(32.*phase); // add high freq

      sum+=sin(tau*phase)*sin(pi*t/(64.*b2t))/1000.;
    }

    dest+=1.6*mix(.2,1.,sidechain)*sum;
  }

  return clip(1.3*tanh(dest));
}
