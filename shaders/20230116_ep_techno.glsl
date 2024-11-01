#define S2T (15.0 / bpm)
#define B2T (60./bpm)
#define T2B (1./B2T)

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

const float TRANSPOSE=3.;
const float SWING = 0.58;

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

float t2sSwing(float t) {
  float st = t / S2T;
  return 2.0 * floor(st / 2.0) + step(SWING, fract(0.5 * st));
}

float s2tSwing(float st) {
  return 2.0 * S2T * (floor(st / 2.0) + SWING * mod(st, 2.0));
}

vec4 seq16(float t, int seq) {
  t = mod(t, 16.0 * S2T);
  int sti = int(t2sSwing(t));
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - prevStepBehind;
  float prevTime = s2tSwing(prevStep);
  float nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + nextStepForward;
  float nextTime = s2tSwing(nextStep);

  return vec4(
    prevStep,
    t - prevTime,
    nextStep,
    nextTime - t
  );
}

vec4 quant(float t, float interval, out float i) {
  interval = max(interval, 1.0);
  float st = t2sSwing(t);

  i = floor(floor(st + 1E-4) / interval + 1E-4);

  float prevStep = ceil(i * interval - 1E-4);
  float prevTime = s2tSwing(prevStep);
  float nextStep = ceil((i + 1.0) * interval - 1E-4);
  float nextTime = s2tSwing(nextStep);

  return vec4(
    prevStep,
    t - prevTime,
    nextStep,
    nextTime - t
  );
}

vec4 quant(float t, float interval) {
  float _;
  return quant(t, interval, _);
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

  float kickt;
  float sidechain;

  { // kick
    float t = kickt=time.x;
    sidechain=smoothstep(0.,.8*B2T,t);

    {
      float env=linearstep(0.3,0.1,t);

      // { // hi pass like
      //   env*=exp(-50.*t);
      // }

      dest+=.5*env*tanh(3.*sin(
        310.*t-20.*exp(-28.*t)
        -20.*exp(-500.*t)
      ));
    }
  }

  { // bass
    float t=time.x;
    vec2 uv=orbit(8.*t)+t*1.3;

    vec2 wave = (
      + sin(320.0 * t)
      + 0.2 * cheapnoiseFBM(4.0 * t, 0.3, 2.0)
    );
    dest+=.4*sidechain*wave;
  }

  { // hihat
    vec4 seq = seq16(time.y, 0xffff);
    float t=seq.t;
    float st=floor(time.y*4.*T2B);
    float decay=exp2(7.-3.*fract(.628*st));
    dest+=.2*tanh(8.*shotgun(5400.*t,1.4,.0))*exp(-decay*t);
  }

  { // clap
    vec4 seq = seq16(time.y, 0x0808);
    float t=seq.t;
    t=lofi(t,1E-4);

    float env=mix(
      exp(-30.*t),
      exp(-200.*mod(t,.013)),
      exp(-80.*max(0.,t-.02))
    );

    vec2 uv=orbit(87.*t)+20.*t;

    dest+=.18*tanh(20.*env*cheapnoiseFBM(8.0 * t, 0.5, 2.0));
  }

  { // ride
    vec4 seq = seq16(time.y, 0xaaaa);
    float t=seq.t;

    float env=mix(
      exp(-5.*t),
      exp(-50.*t),
      .2
    );
    dest+=.1*mix(.3,1.,sidechain)*tanh(10.*shotgun(3000.*t,3.4,.3))*env;
  }

  { // crash
    float t=time.z;
    dest+=.2*mix(.2,1.,sidechain)*tanh(8.*shotgun(4000.*t,3.,.0))*mix(exp(-t),exp(-10.*t),.5);
  }

  { // ep
    vec2 sum=vec2(0);

    for(int i=0;i<28;i++){
      float delay=float(i/7);

      float chord[7]=float[](0.,2.,3.,7.,9.,10.,14.);
      float note=chord[i%7];

      float t = mod(time.z, 64.0 * S2T);
      float st = mod(t2sSwing(t) - 1.0, 64.0);
      st = st < 61.0
        ? lofi(st, 3.0)
        : lofi(st, 2.0);
      st += 1.0;
      st = mod(st, 64.0);

      float prog = step(st, 7.0);

      st += 2.0 * delay;
      st = mod(st, 64.0);

      t = mod(t - s2tSwing(st), 64.0 * S2T);
      float q = (1.4 * S2T) - t;

      vec3 dice=pcg3df(vec3(i, st, 0));
      vec2 dicen=boxMuller(dice.xy);

      float env = exp(-50.0 * max(-q, 0.0));

      float freq=p2f(48.+prog+TRANSPOSE+note+.02*dicen.y);
      float phase=lofi(freq*t+TAU*dice.z,1./32.);
      vec2 wave=vec2(
        +.25*sin(TAU*phase)
        +.25*sin(2.*TAU*phase)
        +.14*sin(3.*TAU*phase)
        +.10*sin(4.*TAU*phase)
      )*vec2(1,-1);

      float zc=linearstep(0.,1E-3,t)*linearstep(0.,1E-3,.75*B2T-t);

      sum+=zc*env*exp(-2.*delay)*wave*r2d(-3.*delay+0.3*dicen.x);
    }

    dest += 0.2 * sum;
  }

  return tanh(1.5*dest);
}
