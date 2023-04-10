#define BPM bpm

#define PI 3.14159265359
#define TAU 6.28318530718

float seed;

float fs( float s ) {
  return fract( sin( s * 114.514 ) * 1919.810 );
}

float random() {
  seed = fs( seed );
  return 2.0 * seed - 1.0;
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0.0 );

  {
    seed = 0.199;

    for ( int i = 0; i < 50; i ++ ) {
      float reltone = 1.0 + random() * 4.0;

      float relfreq = pow( 2.0, reltone );
      float relfreqOt = floor( relfreq + 0.5 );
      float relfreqH = mix( relfreq, relfreqOt, 0.2 );
      reltone = log2( relfreqH ) * 12.0;

      float mtone = reltone;
      float mfreq = 220.0 * pow( 2.0, mtone / 12.0 );

      for ( int j = 0; j < 50; j ++ ) {
        float ptone = mtone + random() * 0.5;

        float freq = 220.0 * pow( 2.0, ptone / 12.0 );

        float noisePhase = TAU * fract( freq * time.z * 5.0 );
        float tt = time.z + pow( time.z / 10.0, 2.0 ) + 0.0001 * sin( TAU / 32.0 * noisePhase );

        vec2 phase = TAU * fract( freq * tt ) + TAU * vec2( random(), random() );
        dest += 0.002 * sin( phase );
      }
    }
  }

  return dest;
}
