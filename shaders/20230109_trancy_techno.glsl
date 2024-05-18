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
#define repeat(i, n) for(int i = 0; i < n; i ++)

const float TRANSPOSE=2.;

const float PI=acos(-1.);
const float TAU=2.*PI;
const float P4=pow(2.,5./12.);
const float P5=pow(2.,7./12.);

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

vec2 cheapnoise(float t) {
  uvec3 s=uvec3(t * 256.0);
  float p=fract(t * 256.0);

  vec3 dice;
  vec2 v = vec2(0.0);

  dice=vec3(pcg3d(s + 0u)) / float(-1u) - vec3(0.5, 0.5, 0.0);
  v += dice.xy * smoothstep(1.0, 0.0, abs(p + dice.z));
  dice=vec3(pcg3d(s + 1u)) / float(-1u) - vec3(0.5, 0.5, 1.0);
  v += dice.xy * smoothstep(1.0, 0.0, abs(p + dice.z));
  dice=vec3(pcg3d(s + 2u)) / float(-1u) - vec3(0.5, 0.5, 2.0);
  v += dice.xy * smoothstep(1.0, 0.0, abs(p + dice.z));

  return 2.0 * v;
}

vec2 cheapnoiseFBM(float t, float pers, float lacu) {
  vec3 sum = vec3(0);

  repeat(i, 5) {
    sum += vec3(cheapnoise(t), 1);
    sum /= pers;
    t *= lacu;
  }

  return sum.xy / sum.z;
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 up = abs(z.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
  vec3 x = normalize(cross(up, z));
  return mat3(x, cross(z, x), z);
}

vec3 cyclic(vec3 p, float pers, float lacu) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  repeat(i, 5) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum /= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
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
    float decay=floor(time.x/(.25*b2t))==2.?10.:90.;
    dest+=.2*tanh(8.*shotgun(8000.*t,1.,.0))*exp(-decay*t);
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

    dest+=.23*tanh(20.*env*cheapnoiseFBM(12.0 * t, 0.5, 2.0));
  }

  // -- ride ---------------------------------------------------------------------------------------
  {
    float t=mod(time.y,.5*b2t);

    dest+=.15*sidechain*tanh(10.*shotgun(4200.*t,2.4,.4))*exp(-10.*t);
  }

  // -- snare909 -----------------------------------------------------------------------------------
  if(inrange(time.z,60.*b2t,64.*b2t)){
    float t=mod(time.x,exp2(-2.-floor(time.y/(2.*b2t)))*b2t);

    vec2 uv=orbit(200.*t)+237.*t;
    vec2 wave=tanh(20.*cheapnoise(30.0 * t));
    wave+=sin(1400.*t-exp(-t*80.)*30.);

    float amp=.14*smoothstep(-4.*b2t,4.*b2t,time.y);
    dest+=amp*tanh(2.*wave*exp(-20.*t));
  }

  // -- crash --------------------------------------------------------------------------------------
  {
    float t=time.z;
    dest+=.2*mix(.2,1.,sidechain)*tanh(8.*shotgun(4000.*t,3.,.0))*mix(exp(-t),exp(-10.*t),.5);
  }

  // -- bass ---------------------------------------------------------------------------------------
  {
    float st=mod(floor(time.y/(.25*b2t)),8.);
    float t=mod(mod(time.y,2.*b2t),.75*b2t);
    float len=(t<.5*b2t?0.5:0.25)*b2t;
    t=mod(t,.5*b2t);

    float env=linearstep(0.,.001,t)*linearstep(0.,.01,len-t-.1*b2t);
    float phase=p2f(24.+TRANSPOSE)*t;
    vec2 uv=orbit(phase);
    vec2 wave=mix(
      vec2(tanh(10.*sin(TAU*phase))),
      cyclic(vec3(4.0 * uv, 2.0), 1.0, 2.0).xy,
      0.4
    );
    dest+=.6*sidechain*env*wave;
  }

  // -- chord --------------------------------------------------------------------------------------
  {
    float chord[8]=float[](
      0.,3.,7.,10.,14.,15.,19.,22.
    );
    vec2 sum=vec2(0);

    float t=time.z;

    for(int i=0;i<64;i++){
      float fi=float(i);
      vec3 dice=pcg3df(vec3(i));

      float freq=p2f(48.+TRANSPOSE+chord[i%8]+.4*(dice.x-.5));
      float phase=freq*t;
      vec3 p = vec3(
        0.3 * orbit(phase),
        0.01 * time.w
      );
      vec2 wave = (
        cyclic(p, 0.5, 2.0)
        - cyclic(p + 0.05, 0.5, 2.0)
      ).xy;
      sum+=vec2(wave)*r2d(fi);
    }

    dest+=.2*mix(.2,1.,sidechain)*sum;
  }

  // -- arp ----------------------------------------------------------------------------------------
  {
    float arp[5]=float[](
      0.,2.,3.,7.,10.
    );
    vec2 sum=vec2(0);

    for(int i=0;i<128;i++){
      vec3 dice=pcg3df(vec3(i));
      float delay=float(i/2%4);

      float t=mod(time.x-.5*b2t*delay,.25*b2t);
      float st=floor(time.z/(.25*b2t)-2.*delay);

      float arpseed=fract(.6+.375*mod(st,8.));
      float note=48.+TRANSPOSE;
      note+=arp[int(arpseed*10.)%5];
      note+=12.*floor(arpseed*2.);
      note+=float(i%2)*7.;
      float freq=p2f(note+.3*boxMuller(dice.xy).x);
      float phase=freq*time.z;
      float wave=cheapfiltersaw(phase,mix(.4,.0,exp(-10.*t))+.05*delay)-cheapfiltersaw(phase,.4);
      sum+=vec2(wave)*r2d(TAU*dice.z)*exp(-.5*delay);
    }

    dest+=.06*mix(.2,1.,sidechain)*sum;
  }

  return tanh(1.5*dest);
}
