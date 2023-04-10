#define PI 3.141592654
#define TAU 6.283185307
#define BPM bpm
#define P4 1.33483985417
#define P5 1.49830707688
#define beat *60.0/BPM

#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define lofir(i,m) (floor((i)/(m)+0.5)*(m))
#define saw(p) (2.*fract(p)-1.)
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (pow(2.,((i)-69.)/12.)*440.)
#define fs(i) (fract(sin((i)*114.514)*1919.810))
#define inRange(a,b,x) ((a)<=(x)&&(x)<(b))

const float TRANSPOSE = 2.0;

uniform sampler2D sample_noise;
uniform vec4 sample_noise_meta;
uniform sampler2D image_fbm;

float envA( float t, float a ) {
  return linearstep( 0.0, a, t );
}

float envAR( float t, float l, float a, float r ) {
  return envA( t, a ) * linearstep( l, l - r, t );
}

vec2 orbit( float t ) {
  return vec2( cos( TAU * t ), sin( TAU * t ) );
}

float chords[8] = float[](
  0.0, 0.0, 3.0, 3.0, 7.0, 7.0, 10.0, 14.0
);

vec2 noise( float t ) {
  return sampleSinc( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) );
}

vec2 random2( float t ) {
  return fract( sampleNearest( sample_noise, sample_noise_meta, mod( t, sample_noise_meta.w ) ) );
}

float euclideanRhythmsInteg(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((t-pulses)/pulses)+1.+fract(time);
}

float euclideanRhythmsRest(float pulses,float steps,float time){
  float t=mod(floor(time)*pulses,steps);
  return floor((steps-t-1.)/pulses)+1.-fract(time);
}

vec2 pan(float x){
  return mix(vec2(2,0),vec2(0,2),x);
}

vec2 shotgun(float t,float spread,float snap){
  vec2 sum=vec2(0);
  for(int i=0;i<64;i++){
    float dice=fs(float(i));

    float partial=exp2(spread*dice);
    partial=mix(partial,floor(partial+.5),snap);

    vec2 pan=mix(vec2(2,0),vec2(0,2),fs(dice));
    sum+=sin(TAU*t*partial)*pan;
  }
  return sum/128.;
}

vec2 kick( float t, float freq ) {
  float phase = freq * t - 11.0 * ( exp( -25.0 * t ) + exp( -100.0 * t ) + exp( -700.0 * t ) );
  float fmA = sin( TAU * 1.0 * phase + 1.4 );
  vec2 fmB = 0.5 * exp( -20.0 * t ) * tri( 0.5 * phase + fmA + vec2( 0.2, 0.24 ) );
  return clip( 1.0 * vec2( exp( -4.0 * t ) * sin( TAU * phase + fmB ) ) );
}

vec2 rimshot( float t ) {
  if ( t < 0.0 ) { return vec2( 0.0 ); }
  float attack = exp( -t * 400.0 ) * 0.6;
  vec2 wave = (
    tri( t * 450.0 * vec2( 1.005, 0.995 ) - attack ) +
    tri( t * 1800.0 * vec2( 0.995, 1.005 ) - attack )
  );
  return clip( 4.0 * wave * exp( -t * 400.0 ) );
}

vec2 filterSaw(float freq,float t,float cutoff,float reso){
  vec2 sum=vec2(0);
  for(int i=1;i<=64;i++){
    float fi=float(i);
    float freqp=freq*fi;
    float omega=freqp/cutoff;
    float omegaSq=omega*omega;

    float a=4.0*reso+omegaSq*omegaSq-6.0*omegaSq+1.0;
    float b=4.0*omega*(omegaSq-1.0);
    float cut=1.0/sqrt(a*a+b*b);
    float offset=atan(a,b);

    sum+=0.66*sin(freqp*t*TAU-offset)/fi*cut;
  }
  return sum;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest=vec2(0);

  // kick
  float kickt=mod(time.x,1. beat);
  float sidechain=linearstep(0.,.6 beat,kickt);

  if(inRange(0. beat,61. beat,time.z)){
    dest+=.5*kick(kickt,50.);
  }

  // pad
  {
    vec2 sum=vec2(0);
    float st=mod(floor((time.z)/(.25 beat)),32.);

    const int CHORD_NOTES=5;
    const int UNISON=4;
    int pitchTable[CHORD_NOTES]=int[](0,3,7,10,14);

    for(int i=0;i<128;i++){
      float fi=float(i);

      float windowcenter=48.-48.*sin(TAU*time.z/(8. beat));
      float windowwidth=16.0;
      
      float partialamp=.5+.5*cos(clip((fi-windowcenter)/windowwidth)*PI);
      if(partialamp==0.){continue;}

      float t=time.z;

      float note=float(36+12*(i/UNISON/CHORD_NOTES)+pitchTable[i/UNISON%CHORD_NOTES])+TRANSPOSE;
      float freq=p2f(note)*mix(.99,1.01,fract(fi*.72));
      vec2 pan=pan(fract(fi*.42));

      vec2 uv=vec2(.1*t+.47*fi);
      uv+=.3*orbit(freq*t);
      float osc=texture(image_fbm,uv).x-.5;

      sum+=0.3*sidechain*partialamp*pan*osc; // fbm osc
    }

    dest+=clip(sum);
  }

  return tanh(1.5*dest);
}
