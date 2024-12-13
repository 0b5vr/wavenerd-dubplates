#define transpose -6.0

#define b2t (60./bpm)
#define t2b (1./b2t)
#define zero min(0, int(bpm))

#define saturate(i) clamp(i,0.,1.)
#define clip(i) clamp(i,-1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define repeat(i,n) for(int i=zero;i<(n);i++)

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

  float trans=transpose+(mod(time.z,16.*b2t)<(12.*b2t)?0.:-2.);

  { // kick
    float t=time.x;
    sidechain=smoothstep(0.,1E-3,b2t-t)*smoothstep(0.,.8*b2t,t);

    if(time.z<61.*b2t){
      float env=linearstep(0.3,0.15,t);

      float phase=50.*t-11.*(exp(-25.*t)+exp(-100.*t)+exp(-700.*t));
      phase+=.2*exp(-20.*t)*sin(tau*phase+1.); // fm attack mod
      float wave=sin(tau*phase);

      dest+=.5*env*tanh(2.*wave);
    }
  }

  { // bass
    float t=mod(time.x,.25*b2t);

    float env=exp(-10.*t);

    float note=36.+trans;
    float freq=p2f(note);
    vec2 sum=env*vec2(tanh(sin(tau*freq*t)));

    repeat(i,16){
      vec3 dice=pcg3df(vec3(i,floor(time.x/.25/b2t),0));

      float noteu=note+.5*(dice.y-.5);
      float frequ=p2f(noteu);
      float phaseu=frequ*(t+.014*sin(tau*2.*frequ*t))+dice.z;

      float k=mix(.3,.02,exp(-10.*t));
      sum+=env*vec2(cheapfiltersaw(phaseu,k))*r2d(tau*dice.x+1.)/8.;
    }

    float zc=linearstep(0.,1E-3,t)*linearstep(0.,1E-2,.25*b2t-t);
    dest+=.5*zc*sidechain*tanh(1.5*sum);
  }

  { // hihat
    float t=mod(time.x,.25*b2t);
    float st=floor(time.y/.25/b2t);

    float vel=fract(st*.62+.67);
    float env=exp(-exp2(7.-3.*vel)*t);
    vec2 wave=shotgun(4000.*t,2.,.0);
    dest+=.25*env*sidechain*tanh(8.*wave);
  }

  { // rim
    float t=mod(time.y,.25*b2t);
    float st=floor(time.z/.25/b2t);

    float env=exp(-300.*t);
    dest+=.2*step(.5,fract(st*.71+.4))*env*tanh(4.*(
      +tri(t*400.-.5*env)
      +tri(t*1500.-.5*env)
    ))*vec2(1,-1);
  }

  { // perc
    float t=mod(time.y-1.*b2t,2.*b2t);

    dest+=.2*tanh(5.*shotgun(1100.*t,1.5,.4))*exp(-4.*t);
  }

  { // crash
    float t=time.z;
    float env=mix(exp(-t),exp(-10.*t),.5);
    vec2 wave=shotgun(4000.*t,3.,.0);
    dest+=.3*mix(.2,1.,sidechain)*tanh(8.*wave)*env;
  }

  { // arp
    vec2 sum=vec2(0);
    const float chord[8] = float[](
      0.,0.,3.,3.,7.,7.,10.,14.
    );

    repeat(i,4){
      float fi=float(i);
      float t=mod(time.x,.25*b2t);
      float st=mod(floor((time.z-.75*b2t*fi)/(.25*b2t)),256.);

      float arpseed=fract(.615*st);
      float note=48.+chord[int(arpseed*24.)%8]+12.*floor(arpseed*3.)+trans;
      float freq=p2f(note);
      vec2 phase=t*freq+vec2(.5,0);

      sum+=2.*exp(-10.*t)*exp(-fi)*(
        +fract(phase)-.5
        +step(fract(.2+.5*phase),vec2(.03))
        -step(fract(.7+.5*phase),vec2(.03))
        +fract(p5*phase)-.5
        +step(fract(1.5*phase),vec2(.5))-.5
      )*r2d(time.w);
    }

    dest+=.2*mix(.2,1.,sidechain)*sum;
  }

  { // pad
    vec2 sum=vec2(0);

    const float pitchTable[8]=float[](0.,7.,10.,12.,14.,15.,19.,26.);

    repeat(i,48){
      float fi=float(i);
      vec3 dice=pcg3df(vec3(fi));

      float t=time.z;

      float note=48.+float(pitchTable[i%8])+trans+.1*boxmuller(dice.xy).x;
      float freq=p2f(note);
      float phase=freq*t+dice.z;

      vec3 p1=vec3(2.*orbit(phase),t);
      vec3 p2=vec3(2.*orbit(phase+.05),t);
      vec2 wave=cyclic(p1,2.).xy-cyclic(p2,2.).xy;

      sum+=wave*r2d(fi)/24.;
    }

    dest+=1.*mix(.2,1.,sidechain)*tanh(sum);
  }

  return clip(1.3*tanh(dest));
}
