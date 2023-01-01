#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define TIME2BEAT (BPM/60.)
#define BEAT2TIME (60./BPM)
#define beat *BEAT2TIME
#define P4 1.33483985417
#define P5 1.49830707688

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define inRange(a,b,x) ((a)<=(x)&&(x)<(b))

const float TRANSPOSE=6.;

uniform sampler2D image_fbm;

float zcross( float t, float l, float d ) {
  return linearstep( 0.0, d, t ) * linearstep( 0.0, d, l - t );
}

float sinc(float x){
  return x==0.?1.:sin(TAU*x)/TAU/x;
}

mat2 r2d(float x){
  float c=cos(x),s=sin(x);
  return mat2(c, s, -s, c);
}

vec2 orbit(float t){
  return vec2(cos(TAU*t), sin(TAU*t));
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

float cheapfiltersaw(float phase,float k){
  float wave=mod(phase,1.);
  float c=.5+.5*cos(PI*saturate(wave/(1.-k)));
  return (wave+c-1.)*2.+k;
}

vec2 mainAudio(vec4 time){
  vec2 dest=vec2(0.);

  float sidechain=mix(1.,.3,smoothstep(BEAT2TIME,.0,time.x)*linearstep(.0,.001,time.x));

  // kick
  {
    float t=time.x;

    float env=linearstep(0.25,0.1,t);
    env*=linearstep(.0,.001,t);

    dest+=.6*env*tanh(2.*sin(
      380.*t-35.*exp(-35.*t)
      -20.*exp(-800.*t)
    ));
  }

  // bass
  {
    float t=mod(mod(time.y-.75 beat,2. beat),.75 beat);

    float freq=p2f(24.+TRANSPOSE);
    float phase=freq*t;

    float env=exp(-10.*t);
    float wave=(
      +cheapfiltersaw(phase,.4+.3*env)
      +sin(TAU*phase)
    );
    dest+=.7*sidechain*env*tanh(4.*wave);
  }

  // hihat
  {
    float t=mod(time.x,.25 beat);

    float env=exp(-50.*t);
    env *= linearstep(.0,.001,t);

    vec2 uv=orbit(1100.*t)+100.*t;

    dest+=.7*sidechain*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5);
  }

  // hihat
  {
    float t=mod(time.x-.5 beat,1. beat);

    float env=exp(-10.*t);

    vec2 uv=orbit(800.*t)+137.*t;

    dest+=.4*sidechain*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5);

    uv=orbit(802.*t)+137.*t;

    dest-=.4*sidechain*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5);
  }

  // clap
  {
    float t=mod(time.y-1. beat,2. beat);

    float env=exp(-10.*max(t-.1,0.));
    env*=1.+sinc(20.*t);

    vec2 uv=orbit(123.7*t)+200.*t;

    dest+=.5*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.05).x
    )-.5);
  }

  // ride
  {
    float t=mod(time.x,.5 beat);

    float env=exp(-5.*t);
    env *= linearstep(.0,.001,t);

    vec2 uv=vec2(37.,73.)*t;
    uv=.7*vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )+.3*exp(-5.*t)*orbit(2200.0*t)+1500.0*t;

    dest+=.3*sidechain*env*(vec2(
      texture(image_fbm,uv).x,
      texture(image_fbm,uv+.5).x
    )-.5);
  }

  // rim
  {
    float t=mod(mod(time.z, 2.75 beat),.5 beat);

    float env=exp(-300.*t);

    dest+=.24*env*tanh(4.*(
      +tri(t*400.*vec2(.98,.99)-.5*env)
      +tri(t*1500.*vec2(.99,.98)-.5*env)
    ))*vec2(1,-1);
  }

  // chord
  {
    vec2 sum=vec2(0.);

    int pitchTable[3]=int[](0,3,7);

    for(int i=0;i<96;i++){
      float fi=float(i);
      vec3 dice=pcg3df(vec3(i));

      float delay=floor(dice.x*3.);
      float t=mod(time.y-2.75 beat,4. beat)-delay*.75 beat;

      float freq=p2f(36.+TRANSPOSE+float(pitchTable[i%3]))*mix(.997,1.003,dice.y);
      float offu=dice.z;

      vec2 uv=vec2(.5);
      uv+=.3*exp(-4.*t)*(orbit(freq*t+offu)+.1*orbit(freq*13.*t+offu));
      float tex=tanh(2.*(texture(image_fbm,uv).x-texture(image_fbm,uv+.1).x));

      float ampenv=exp(-3.*max(t-.1,0.));
      ampenv*=linearstep(.0,.001,t);
      float amp=.15*ampenv*sidechain*exp(-delay);
      sum+=vec2(amp*tex)*r2d(70.0*dice.z);
    }

    dest+=sum;
  }

  // riff
  {
    vec2 sum=vec2(0);
    float scale[5]=float[](0.,3.,2.,5.,10.);

    for(int i=0;i<4;i++){
      float fi=float(i);

      float tp=mod(time.z-fi*.5 beat,8. beat);
      float t=mod(mod(tp,1.25 beat),.5 beat);
      float l=mod(tp,1.25 beat)<1. beat?.5 beat:.25 beat;
      float st=floor((tp-t)*4.*TIME2BEAT+.5);

      float arpseed=fract(.645*st);
      float pitch=scale[int(10.*arpseed)%5]+12.*floor(2.*arpseed);
      float freq=p2f(60.+pitch+TRANSPOSE);
      float phase=freq*t;
      float env=exp(-20.*t);
      float dir=mod(st,2.)*2.-1.;
      vec2 wave=orbit(phase*dir);
      sum+=env*exp(-fi)*zcross(t,l,1E-3)*wave*r2d(fi+time.z+st);
    }

    dest+=.2*sidechain*sum;
  }

  return tanh(1.5*dest);
}
