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

mat2 r2d(float x){
  float c=cos(x),s=sin(x);
  return mat2(c,s,-s,c);
}

float tmod(vec4 time, float d) {
  vec4 t = mod(time, timeLength);
  float offset = lofi(t.z - t.x + timeLength.x / 2.0, timeLength.x);
  offset -= lofi(t.z, d);
  return t.x + offset;
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

  // { // kick
  //   float t=time.x;
  //   sidechain=smoothstep(0.,1E-3,b2t-t)*smoothstep(0.,.8*b2t,t);

  //   {
  //     float env=linearstep(0.4,0.15,t);
  //     env*=mix(1.,exp(-50.*t),p3);

  //     dest+=.6*env*tanh(2.*sin(
  //       300.*t-20.*exp(-40.*t)
  //       -5.*exp(-400.*t)
  //     ));
  //   }
  // }

  // { // sub kick
  //   float t=mod(time.x-.25*b2t,.25*b2t);

  //   float zc=linearstep(0.,.002,t)*linearstep(0.,.002,.25*b2t-t);
  //   float env=exp(-10.*t);
  //   float wave=sin(310.*t-2.*exp(-80.*t));
  //   dest+=.5*sidechain*zc*env*wave;
  // }

  // { // low freq noise
  //   float t=time.x;

  //   vec2 wave=cyclic(vec3(200.*t),3.).xy;
  //   dest+=.14*sidechain*wave;
  // }

  // { // hihat
  //   float t=mod(time.x-.5*b2t,1.*b2t);
  //   float env=exp(-20.*t);
  //   dest+=.18*env*tanh(8.*shotgun(5400.*t,1.4,.0));
  // }

  // { // ride
  //   float t=mod(time.y,.5*b2t);

  //   dest+=.1*sidechain*tanh(10.*shotgun(3200.*t,3.4,.1))*exp(-10.*t);
  // }

  // { // perc
  //   float tp=mod(time.y,2.*b2t);
  //   float t=mod(mod(tp,.75*b2t),.5*b2t);
  //   float st=(tp-t)*4.*t2b;

  //   float tone=fract(.3+st*.422);
  //   vec2 wave=cyclic(
  //     32.*vec3(orbit(exp2(6.+2.*tone)),exp2(4.+3.*tone)*t),
  //     1.5
  //   ).xy;

  //   float env=mix(
  //     exp(-30.*t),
  //     exp(-5.*t),
  //     0.2
  //   );
  //   dest+=.4*sidechain*env*tanh(2.*wave);
  // }

  // { // clav
  //   float t=mod(mod(time.y,2.25*b2t),.5*b2t);

  //   float wave=sin(17000.*t);
  //   dest+=.2*exp(-t*200.)*vec2(wave)*r2d(1.4);
  // }

  // { // rim
  //   float t=mod(mod(time.y-.25*b2t,1.25*b2t),.5*b2t);

  //   float env=exp(-300.*t);
  //   dest+=.3*env*tanh(4.*(
  //     +tri(t*400.-.5*env)
  //     +tri(t*1500.-.5*env)
  //   ))*vec2(1,-1);
  // }

  // { // noise
  //   float t=mod(time.z,32.*b2t);
  //   float tt=500.0*exp(-.2*t);
  //   float tt2=500.0*exp(-.2*(t+.00005*(1.-exp(-.2*t))));
  //   vec2 wave=cheapnoise(tt)-cheapnoise(tt2);
  //   dest+=.1*sidechain*wave;
  // }

  { // dual vco
    vec2 sum=vec2(0);

    repeat(i,7){
      float fi=float(i);

      const float freqs[3]=float[](560.,1200.,240.);
      const float times[3]=float[](.25,.75,1.5);

      repeat(j,3){
        float t=tmod(time-times[j]*b2t-.5*b2t*fi,2.*b2t);
        vec2 wave=vec2(0);
        wave+=sin(tau*freqs[j]*t);
        wave+=0.0*sin(2.0*tau*freqs[j]*t+wave); // osc 2
        sum+=exp(-30.*max(t-.05,0.))*exp(-2.*fi)*wave*r2d(fi+10.*time.w);
      }
    }

    dest+=.1*sum;
  }

  return clip(1.3*tanh(dest));
}
