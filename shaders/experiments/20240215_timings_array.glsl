#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define saturate(x) clamp(x, 0., 1.)
#define repeat(i, n) for (int i = 0; i < n; i++)

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);
  float sidechain;

  { // kick
    float tr = time.y;
    float br = tr / B2T;

    float timings[] = float[](0.0, 1.5, 2.0, 3.0, 4.0);
    float s, l;
    repeat(i, 4) {
      if (br < timings[i + 1]) {
        s = timings[i] * B2T;
        l = timings[i + 1] * B2T - s;
        break;
      }
    }

    float t = tr - s;
    float q = l - t;

    sidechain = 0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, q);

    float env = smoothstep(0.0, 0.001, q) * smoothstep(0.3, 0.1, t);

    float wave = sin(
      270.0 * t
      - 30.0 * exp(-t * 20.0)
      - 20.0 * exp(-t * 60.0)
      - 10.0 * exp(-t * 500.0)
    );
    dest += 0.6 * tanh(2.0 * env * wave);
  }

  return tanh(dest);
}
