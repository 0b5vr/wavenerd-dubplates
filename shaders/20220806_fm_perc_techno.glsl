#define BPM bpm
#define beat *(60./bpm)

#define PI 3.14159265359
#define TAU 6.28318530718
#define saturate(i) clamp(i, 0.,1.)
#define clip(i) clamp(i, -1.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define lofi(i,m) (floor((i)/(m))*(m))
#define pwm(x,d) (step(fract(x),(d))*2.0-1.0)
#define tri(p) (1.-4.*abs(fract(p)-0.5))

vec2 orbit(float t){
  return vec2(cos(TAU*t), sin(TAU*t));
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  // time=mod(time,1.0 beat)+0.0 beat;
  // time=mod(time,0.75 beat)+0.0 beat;

  float kickt = mod( time.x, 1.0 beat );
  float sidechain = exp2( mix( -1.5, 0.0, linearstep( 0.0, 1.0 beat, kickt ) ) );

  // kick
  {
    float t = kickt;

    float env = exp( -10.0 * max( 0.0, t - 0.1 ) );

    float phase = 50.0 * t - 10.0 * exp( -30.0 * t );

    dest += 0.6 * env * tri( phase );
  }

  // hihat
  {
    float t = mod( time.x, 0.25 beat );
    float st = mod( floor( time.z / ( 0.25 beat ) ), 16.0 );

    float envptn = fract( 0.4 + 0.651 * st );
    float env = exp( -exp2( 5.0 + 3.0 * envptn ) * t );

    vec2 phase = (
      8000.0 * t // base pitch
      + 1.0 * exp( -1.0 * t ) * tri( 15800.0 * t ) // fm
    ) + vec2( 0.0, 0.05 );

    dest += 0.2 * sidechain * env * pwm( phase, vec2( 0.125 ) );
  }

  // perc
  {
    float t=mod(time.y-.75 beat,2. beat);

    float env=exp(-10.*t);

    dest += 0.2*sidechain*env*tri(
      700.0 * t // base pitch
      - 20.0 * exp( -200.0 * t ) // pitch env
      + 2.0 * tri( 88.0 * t ) // fm
      +vec2(.2,0)
    );
  }

  // perc
  {
    float t = mod( time.x, 0.25 beat );
    float st = mod( floor( time.z / ( 0.25 beat ) ), 16.0 );

    float envptn = fract( 0.4 + 0.631 * st );
    float env = exp( -exp2( 3.0 + 5.0 * envptn ) * t );

    float fmfreqptn = fract( 0.4 + 0.622 * st );
    float fmfreq = exp2( 8.0 + 6.0 * fmfreqptn );

    dest += 0.2 * sidechain * env * tri(
      100.0 * t // base pitch
      - 20.0 * exp( -50.0 * t ) // pitch env
      + 300.0 / fmfreq * tri( fmfreq * t ) // fm
    );
  }

  // woodblock
  {
    float t = mod( time.y - 0.5 beat, 2.0 beat );

    float env = exp( -40.0 * t );

    vec2 phase = (
      400.0 * t // base pitch
      + 0.3 * exp( -1.0 * t ) * tri( 1558.0 * t ) // fm
    ) + vec2( 0.0, 0.1 );

    dest += 0.3 * sidechain * env * tri( phase );
  }

  // ride
  {
    float t = mod( time.x - 0.5 beat, 1.0 beat );

    float env = exp( -5.0 * t );

    vec2 phase = (
      12000.0 * t // base pitch
      + 1.0 * exp( -2.0 * t ) * tri( 18743.0 * t ) // fm
    ) + vec2( 0.0, 0.2 );

    dest += 0.14 * sidechain * env * pwm( phase, vec2( 0.875 ) );
  }

  // bass
  {
    float t = mod( time.x, 0.25 beat );
    float st = mod( floor( time.z / ( 0.25 beat ) ), 16.0 );

    float env = linearstep( 0.01, 0.0, t - 0.22 beat );

    float fmfreqptn = fract( 0.4 + 0.622 * st );
    float fmfreq = exp2( 4.0 + 6.0 * fmfreqptn );

    dest += 0.2 * sidechain * env * (
      tri(
        50.0 * t // base pitch
        + 0.2 * tri( fmfreq * t ) // fm
      )
      + tri( 50.0 * t )
    );
  }

  // hit
  {
    float t = mod( time.y - 1.25 beat, 2.0 beat );

    float env = exp( -14.0 * t );

    vec2 phase = (
      800.0 * t // base pitch
      + 5.0 * tri( 20.0 * t ) // fm
    ) + vec2( 0.0, 0.5 );

    dest += 0.25 * sidechain * env * pwm( phase, vec2( 0.75 ) );
  }

  return tanh(2.*dest);
}
