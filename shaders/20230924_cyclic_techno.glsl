#define BPM bpm
#define B2T (60.0 / bpm)

#define saturate(i) clamp(i,0.,1.)
#define clip(i) clamp(i,-1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define repeat(i,n) for(int i=0;i<(n);i++)
#define lofi(i,m) (floor((i)/(m))*(m))
#define tri(p) (1.-4.*abs(fract(p)-0.5))
#define p2f(i) (exp2(((i)-69.0) / 12.0) * 440.0)

const float TRANSPOSE = 4.0;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;
const uint UINT_MAX = 0xffffffffu;

uvec3 hash3(uvec3 v) {
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

vec3 hash3f(vec3 v) {
  uvec3 r = hash3(floatBitsToUint(v));
  return vec3(r) / float(UINT_MAX);
}

vec2 cis(float t) { return vec2(cos(t), sin(t)); }

mat2 rotate2D(float t) { return mat2(cos(t), sin(t), -sin(t), cos(t)); }

vec2 boxMuller(vec2 xi) {
  float r = sqrt(-2.0 * log(xi.x));
  float t = xi.y;
  return r * cis(TAU * t);
}

vec2 shotgun( float t, float spread ) {
  vec2 sum = vec2(0);
  repeat(i, 64) {
    vec3 dice = hash3f(vec3(i));

    float partial = exp2(spread * dice.x);

    sum += vec2(sin(TAU * t * partial)) * rotate2D(TAU * dice.y);
  }
  return sum / 64.;
}

float cheapFilterSaw( float phase, float k ) {
  float wave = mod( phase, 1.0 );
  float c = smoothstep( 1.0, 0.0, wave / k );
  return ( wave + c ) * 2.0 - 1.0 - k;
}

mat3 orthBas(vec3 z){
  z=normalize(z);
  vec3 x=normalize(cross(vec3(0,1,0),z));
  vec3 y=cross(z,x);
  return mat3(x,y,z);
}

vec3 cyclicNoise(vec3 p, float lacu, float pers) {
  vec4 sum = vec4(0.0);
  mat3 basis = orthBas(vec3(2.0, -3.0, 1.0));
  float amp = 1.0;

  repeat(i, 5) {
    p *= basis;

    p += sin(p.yzx);
    sum += amp * vec4(cross(cos(p), sin(p.zxy)), 1.0);

    amp *= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
}

vec4 seq16( int seq, float st ) {
  st = mod( st, 16.0 );
  int sti = int( st ) % 16;
  int rotated = ((seq >> (15 - sti)) | (seq << (sti + 1))) & 0xffff;

  float prevStepBehind = log2(float(rotated & -rotated));
  float prevStep = float(sti) - prevStepBehind;
  float nextStepForward = 16.0 - floor(log2(float(rotated)));
  float nextStep = float(sti) + nextStepForward;

  return vec4(prevStep, st - prevStep, nextStep, nextStep - st);
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  float sidechain = 1.0;

  // kick
  {
    float t = time.x;
    sidechain = smoothstep( 0.0, B2T, t );

    float env = linearstep( 0.3, 0.1, t );

    dest += 0.5 * env * tanh( 3.0 * sin(
      340.0 * t
      -25.0 * exp( -20.0 * t )
      -35.0 * exp( -70.0 * t )
      -20.0 * exp( -500.0 * t )
    ) );
  }

  { // bass
    float t = mod( time.x, B2T );
    float q = B2T - t;

    float env = linearstep( 0.0, 0.002, t ) * linearstep( 0.0, 0.002, q );

    vec2 sum = vec2( sin( 280.0 * t ) );

    repeat( i, 4 ) {
      vec3 dice = hash3f( vec3( i + 18 ) );
      vec2 dicei = boxMuller( dice.xy );
      float phase = ( 280.0 + 10.0 * dice.x + TAU * dice.z ) * t;
      vec3 pos = vec3( cis( phase ), 8.0 + 4.0 * t / B2T );
      vec2 wave = cyclicNoise( pos, 2.0, 0.5 ).xy * rotate2D( float( i ) );
      sum += 0.3 * wave;
    }

    dest += 0.5 * sidechain * env * tanh( 3.0 * sum );
  }

  { // hihat
    float l = 0.25 * B2T;
    float t = mod( time.x, l );
    float q = l - t;

    float env = mix(
      exp( -6.0 * t ),
      exp( -40.0 * t ),
      0.94
    );
    dest += 0.24 * env * mix( 1.0, sidechain, 0.9 ) * tanh( 8.0 * shotgun( 4000.0 * t, 1.8 ) );
  }

  { // clap
    float t = mod( time.y - B2T, 2.0 * B2T );

    float env = mix(
      exp( -14.0 * t ),
      exp( -200.0 * mod( t, 0.015 ) ),
      exp( -90.0 * max( 0.0, t - 0.02 ) )
    );

    vec3 pos = vec3( 4.0 * cis( 900.0 * t ), 240.0 * t );
    vec2 wave = cyclicNoise( pos, 2.0, 0.5 ).xy;
    dest += 0.1 * tanh( 12.0 * env * wave );
  }

  { // perc
    repeat( i, 16 ) {
      vec3 seed = vec3( 24.24, 21.61, 41.37 );
      vec3 dice = hash3f( vec3( i ) + seed );
      vec3 dice2 = hash3f( dice );
      vec2 dicei = boxMuller( dice2.xy );

      float t = mod(
        mod( time.z, 16.0 * B2T ) - 0.25 * B2T * float( i ),
        0.25 * B2T * floor( 3.0 + 14.0 * dice.x )
      );

      float env = exp( -exp2( 2.0 + 6.0 * dice.y ) * t );
      float amp = 0.15 * exp2( dice.z );

      vec2 wave = cyclicNoise( vec3(
        exp2( -2.0 + 4.0 * dice2.z ) * cis( exp2( 8.0 + 4.0 * dicei.x ) * t ) + 20.0 * dice2.xy,
        exp2( 7.0 + 7.0 * dicei.y ) * t
      ), 2.0, 0.5 ).xy * rotate2D( float( i ) );

      dest += amp * sidechain * tanh( 3.0 * env * wave );
    }
  }

  { // crash
    float t = time.z;

    float env = mix(
      exp( -t ),
      exp( -10.0 * t ),
      0.5
    );

    vec2 wave = shotgun( 4000.0 * t, 3.0 );

    dest += 0.2 * mix( 0.2, 1.0, sidechain ) * tanh( 8.0 * wave ) * env;
  }

  return tanh( dest );
}
