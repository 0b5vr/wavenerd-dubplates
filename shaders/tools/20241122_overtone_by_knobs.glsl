const float PI = acos(-1.0);
const float TAU = 2.0 * PI;

uniform vec4 param_knob0;
uniform vec4 param_knob1;
uniform vec4 param_knob2;
uniform vec4 param_knob3;
uniform vec4 param_knob4;
uniform vec4 param_knob5;
uniform vec4 param_knob6;
uniform vec4 param_knob7;

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0.0);

  float t = time.y;
  float q = timeLength.y - t;

  float env = smoothstep(0.0, 0.1, t) * smoothstep(0.0, 0.1, q);

  float phase = TAU * 220.0 * t;

  dest += paramFetch(param_knob0) * sin(1.0 * phase);
  dest += paramFetch(param_knob1) * sin(2.0 * phase);
  dest += paramFetch(param_knob2) * sin(3.0 * phase);
  dest += paramFetch(param_knob3) * sin(4.0 * phase);
  dest += paramFetch(param_knob4) * sin(5.0 * phase);
  dest += paramFetch(param_knob5) * sin(6.0 * phase);
  dest += paramFetch(param_knob6) * sin(7.0 * phase);
  dest += paramFetch(param_knob7) * sin(8.0 * phase);

  return 0.1 * env * dest;
}
