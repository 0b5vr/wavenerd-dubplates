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

const float SWING = 0.5;

const float pi=acos(-1.);
const float tau=2.*pi;
const float p4=exp2(5./12.);
const float p5=exp2(7./12.);
const uint uint_max=0xffffffffu;

uniform vec4 param_knob3; // kick cut

#define p3 paramFetch(param_knob3)

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

float t2sSwing(float t) {
  float st = 4.0 * t / b2t;
  return 2.0 * floor(st / 2.0) + step(SWING, fract(0.5 * st));
}

float s2tSwing(float st) {
  return 0.5 * b2t * (floor(st / 2.0) + SWING * mod(st, 2.0));
}

vec4 seq16(float t, int seq) {
  t = mod(t, 4.0 * b2t);
  int sti = clamp(int(t2sSwing(t)), 0, 15);
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float i_prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - i_prevStepBehind;
  float prevTime = s2tSwing(prevStep);
  float i_nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + i_nextStepForward;
  float nextTime = s2tSwing(nextStep);

  return vec4(
    prevStep,
    t - prevTime,
    nextStep,
    nextTime - t
  );
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

  float trans=transpose+(mod(time.z,16.*b2t)<(12.*b2t)?0.:-0.);

  float sidechain=1.;

  { // kick
    vec4 seq=seq16(time.y, 0x9292);
    float t=seq.t;
    float q=seq.q;
    sidechain=smoothstep(0.,1E-3,q)*smoothstep(0.,.8*b2t,t);

    {
      float env=linearstep(.0,.001,t)*linearstep(.0,.01,q)*linearstep(0.3,0.1,t);
      env *= mix(1.0, exp2(-90.0 * t), p3);

      float wave=mix(
        sin(300.*t-65.*exp(-80.*t)),
        sin(200.*t-15.*exp(-40.*t)),
        step(60.*b2t,time.z)
      );
      dest+=.6*tanh(3.*env*wave);
    }
  }

  { // bass
    vec4 seq=seq16(time.y, 0x8080);
    float t=seq.t;

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

  // { // hihat
  //   vec4 seq=seq16(time.y, 0xffff);
  //   float t=seq.t;

  //   float env=exp(-40.*t);
  //   env*=linearstep(.0,.001,t);

  //   vec2 wave=cyclic(vec3(6000.*t),1.2).xy;

  //   dest+=.4*sidechain*env*tanh(5.*wave);
  // }

  // { // rim
  //   vec4 seq=seq16(time.y, 0x6b6b);
  //   float t=seq.t;

  //   float env=exp(-300.*t);
  //   dest+=.3*env*tanh(4.*(
  //     +tri(t*400.-.5*env)
  //     +tri(t*1500.-.5*env)
  //   ))*vec2(1,-1);
  // }

  { // fm perc
    vec4 seq=seq16(time.y, 0xffff);
    float t=seq.t;
    t=lofi(t,1E-4);
    float st=floor(time.z/.25/b2t);
    vec3 dice=pcg3df(vec3(st));

    float freq=exp2(8.+3.*dice.x);
    float env=exp(-exp2(3.+5.*dice.y)*t);
    float fm=env*exp2(3.+3.*dice.z)*sin(freq*exp(-t));
    float wave=sin(fm);
    dest+=.1*mix(.2,1.,sidechain)*vec2(wave)*r2d(st);
  }

  // { // hihat 2
  //   vec4 seq=seq16(time.y, 0xffff);
  //   float t=seq.t;
  //   float st=seq.s;

  //   float env=exp(-exp2(3.+2.*fract(.4+.628*st))*t);
  //   env*=linearstep(.0,.001,t);

  //   vec2 wave=shotgun(4000.*t,3.,.5);

  //   dest+=.4*sidechain*env*tanh(5.*wave);
  // }

  // { // clap
  //   vec4 seq=seq16(time.y, 0x0008);
  //   float t=seq.t;

  //   float env=exp(-40.*t)+.02*exp(-5.*t);

  //   t+=0.1*sin(t*90.0);
  //   vec3 p=vec3(10.*orbit(59.8*t),+250.*t);
  //   vec2 wave=cyclic(p,2.).xy;

  //   dest+=.2*tanh(20.*env*wave);
  // }

  // { // s4 transition - arp
  //   vec2 sum=vec2(0);
  //   const float chord[8] = float[](
  //     0.,0.,3.,3.,7.,7.,10.,14.
  //   );

  //   repeat(i,4){
  //     float fi=float(i);
  //     float t=mod(time.x,.25*b2t);
  //     float st=mod(floor((time.z-.75*b2t*fi)/(.25*b2t)),256.);

  //     float arpseed=fract(.615*st);
  //     float note=48.+chord[int(arpseed*24.)%8]+12.*floor(arpseed*3.)+trans;
  //     float freq=p2f(note);
  //     vec2 phase=t*freq+vec2(.5,0);

  //     sum+=2.*exp(-10.*t)*exp(-fi)*(
  //       +fract(phase)-.5
  //       +step(fract(.2+.5*phase),vec2(.03))
  //       // -step(fract(.7+.5*phase),vec2(.03))
  //       // +fract(p5*phase)-.5
  //       // +step(fract(1.5*phase),vec2(.5))-.5
  //     )*r2d(time.w);
  //   }

  //   dest+=.0*.2*mix(.2,1.,sidechain)*sum;
  // }

  // { // s4 transition - snareroll
  //   float roll=mix(.25*b2t,.125*b2t,step(56.*b2t,time.z));
  //   float t=mod(time.x,roll);

  //   vec3 p=vec3(10.*orbit(500.*t),1000.*t);
  //   vec2 wave=exp(-15.*t)*cyclic(p,2.).xy;
  //   float phase=200.*t-3.*exp(-140.*t);
  //   float sine=sin(tau*phase+.1*sin(.33*tau*phase));
  //   sine*=(1.2+.4*sine);
  //   wave+=exp(-30.*t)*sine;

  //   float amp=linearstep(0.,60.*b2t,time.z);
  //   dest+=.3*amp*amp*mix(.5,1.,sidechain)*tanh(3.*wave);
  // }

  // { // s4 transition - sweep
  //   float prog=time.z/60./b2t;

  //   vec2 sum=vec2(0);
  //   repeat(i,20){
  //     float fi=float(i);

  //     float t=time.z;
  //     t-=.003*(1.-prog)*fi;
  //     vec3 p=vec3(5.*orbit(mix(60.,50.,prog)*t),1.*t);
  //     vec2 wave=cyclic(p,.5).xy;
  //     sum+=wave*r2d(tau*fi/20.);
  //   }

  //   dest+=.07*prog*prog*sidechain*sum;
  // }

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
