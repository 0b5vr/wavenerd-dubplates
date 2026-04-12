#define BPS ( bpm / 60.0 )

const float TAU = 2.0 * acos( -1.0 );

uniform vec4 param_knob0;

uvec3 hash3u( uvec3 v ) {
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

vec3 hash3f( vec3 v ) {
  uvec3 x = floatBitsToUint( v );
  return vec3( hash3u( x ) ) / float( -1u );
}

vec2 smoother( vec2 t ) {
  return ( t * t * t * ( t * ( t * 6.0 - 15.0 ) + 10.0 ) );
}

vec2 orbit( float t ) {
  return vec2( cos( TAU * t ), sin( TAU * t ) );
}

vec2 getDir( ivec2 p ) {
    return orbit( hash3f( vec3( p & 255, 0 ) ).x );
}

float perlin2d( vec2 p ) {
  vec2 cell = floor( p );
  vec2 cellCoord = p - cell;
  ivec2 cellIndex = ivec2( cell );

  vec2 cellCoordS = smoother( cellCoord );

  return mix(
    mix(
      dot( getDir( cellIndex ), cellCoord ),
      dot( getDir( cellIndex + ivec2( 1, 0 ) ), cellCoord - vec2( 1.0, 0.0 ) ),
      cellCoordS.x
    ),
    mix(
      dot( getDir( cellIndex + ivec2( 0, 1 ) ), cellCoord - vec2( 0.0, 1.0 ) ),
      dot( getDir( cellIndex + ivec2( 1, 1 ) ), cellCoord - 1.0 ),
      cellCoordS.x
    ),
    cellCoordS.y
  );
}

float fbm( vec2 p ) {
  return (
    + perlin2d( 2.0 * p ) / 2.0
    + perlin2d( 4.0 * p ) / 4.0
    + perlin2d( 8.0 * p ) / 8.0
    + perlin2d( 16.0 * p ) / 16.0
  );
}

vec2 mainAudio( vec4 time ) {
  vec2 dest = vec2( 0 );

  float t = time.x;
  if ( mod( time.z, 16.0 / BPS ) > 15.0 / BPS ) {
    t = mod( t, 0.5 / BPS );
  }
  float sidechain = smoothstep( 0.3 / BPS, 0.6 / BPS, t );

  // subbass
  float phase = 300.0 * t - 60.0 * exp( -40.0 * t );
  float subbass = sin( phase );

  // kick
  float p = 1.0 + exp( -mix( 6.0, 1.2, paramFetch( param_knob0 ) ) );

  dest += tanh( 30.0 * subbass );

  // unison x3
  t *= p;
  phase = 300.0 * t - 60.0 * exp( -40.0 * t );

  dest += tanh( 30.0 * (
    sin( phase + 0.2 * sin( 3.0 * phase ) - 0.1 * sin( 13.1 * phase ) )
  ) );

  t *= p;
  phase = 300.0 * t - 60.0 * exp( -40.0 * t );

  dest += tanh( 30.0 * (
    sin( phase + 0.2 * sin( 3.0 * phase ) - 0.1 * sin( 13.1 * phase ) )
  ) );

  t *= p;
  phase = 300.0 * t - 60.0 * exp( -40.0 * t );

  dest += tanh( 30.0 * (
    sin( phase + 0.2 * sin( 3.0 * phase ) - 0.1 * sin( 13.1 * phase ) )
  ) );

  // hihat
  t = mod( time.x - 0.5 / BPS, 1.0 / BPS );

  float env = exp( -20.0 * t );

  vec2 uv = orbit( 800.0 * t ) + orbit( 4000.0 * t ) * exp( -100.0 * t ) + 137.0 * t;

  dest += 0.5 * sidechain * env * tanh( 5.0 * vec2(
    fbm( uv ),
    fbm( uv - 0.5 )
  ) );

  uv = orbit( 802.0 * t ) + orbit( 4000.0 * t ) * exp( -100.0 * t ) + 137.0 * t;

  dest -= 0.5 * sidechain * env * tanh( 5.0 * vec2(
    fbm( uv ),
    fbm( uv - 0.5 )
  ) );

  // scream
  t = time.y;

  uv = 2.0 * orbit( 79.0 * t ) + 0.2 * orbit( 2000.0 * t ) + 10.0 * t;

  dest += 0.2 * sidechain * tanh( 20.0 * vec2(
    fbm( uv ),
    fbm( uv + 0.05 )
  ) );

  // distort master
  return mix( tanh( 4.0 * dest ), vec2( subbass ), 0.3 );
}
