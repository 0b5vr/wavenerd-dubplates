const float PI = acos(-1.0);
const float TAU = 2.0 * PI;

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float t = time.y;
  float q = timeLength.y - t;

  float env = smoothstep(0.0, 0.1, t) * smoothstep(0.0, 0.1, q);

  float phase = 440.0 * t;

  float sine = sin(TAU * phase);
  float saw = 2.0 * fract(phase) - 1.0;
  float tri = 1.0 - 4.0 * abs(fract(phase + 0.25) - 0.5);
  float square = 2.0 * step(0.5, fract(phase)) - 1.0;

  float kind = floor(mod(time.z / timeLength.y, 4.0));
  dest += mix(
    mix(
      sine,
      saw,
      step(1.0, kind)
    ),
    mix(
      tri,
      square,
      step(3.0, kind)
    ),
    step(2.0, kind)
  );

  return 0.5 * env * dest;
}
