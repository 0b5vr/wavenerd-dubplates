//                                                                                   
//    ▀██                            ▄█▀                             ▀█▄             
//   ▄▄██  ▄▄ ▄▄▄   ▄▄▄▄   ▄▄   ▄▄  ██  ▄▄▄▄▄  ▄▄▄  ▄   ▄ ▄   ▄  ▄▄▄   ██  ██        
// ▄█▀ ██   ██▀ ██  ▄▄▄██  ██ ▄ ██  ██    █   █   █ █▄▄▀   ▀▄▀  █ ▄ █  ██            
// ██  ██   ██     ██  ██  ██ █ ██  ██    █   ▀▄▄▄▀ █  ▀▄   █   ▀▄▄▄▀  ██  ▄▄        
// ▀█▄▄▀█▄ ▄██▄    ▀█▄▄▀█▄ ▀██▀██▀   ▀█▄                             ▄█▀  ▄█▀        
//                                                                                   
// Audio/Visual Event "draw(tokyo); #2"                                              
// CIRCUS TOKYO @ Shibuya, Tokyo                           2025-03-22 14:00 - 20:30  
//                                                                                   
//                            0b5vr x Renard x ukonpower                             

// Shoutouts to:
// - 🔥 Renard and ukonpower
// - 🤲 Saina, W0NYV, and all the draw(); crew
// - ☄️ Revision and Meteoriks
// - 🗼 Tokyo Demo Fest and SESSIONS
// - 🔊 Demoscene techno headz! mfx, Epoch, luchak, chlumpie, dok & jon
// - 🗜️ The 64k scene! Conspiracy, farbrausch, Logicoma, Mercury
// - ✨️ The 4k scene! 0x4015, iq, LJ & Virgill, Blueberry, pestis
// - 💻️ Atsushi Tadokoro, moistpeace, yaxu, Algorave
// - 🎛️ Yebisu303, Nyolfen, fendoap, DJ SHARPNEL, Elektron, Dirtywave

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define repeat(i, n) for (int i = ZERO; i < n; i++)

const float PI = acos(-1.0);
const float TAU = PI * 2.0;

uniform vec4 param_knob7; // oidos filter

#define p7 paramFetch(param_knob7)

uvec3 hash3u(uvec3 v) {
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
  uvec3 x = floatBitsToUint(v);
  return vec3(hash3u(x)) / float(-1u);
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  { // oidos drone
    vec2 sum = vec2(0.0);

    float lpfFreq = exp2(mix(
      log2(20.0),
      log2(22000.0),
      linearstep(0.0, 0.5, p7)
    ));
    float hpfFreq = exp2(mix(
      log2(20.0),
      log2(22000.0),
      linearstep(0.5, 1.0, p7)
    ));

    repeat(i, 2500) {
      vec3 diceA = hash3f(vec3(i / 50, 65, 76));
      vec3 diceB = hash3f(vec3(i));

      float t = mod(time.z - diceA.x * (64.0 * B2T), 64.0 * B2T);
      float env = sin(PI * t / (64.0 * B2T));

      float tone = 7.0 + 8.0 * diceA.y + 0.05 * diceB.y;
      float freq = exp2(tone);
      vec2 phase = t * freq + fract(diceB.xy * 999.0);

      float amp = 1.0;
      const float FILTER_WIDTH = 0.3;
      amp *= smoothstep(
        1.0 + FILTER_WIDTH,
        1.0,
        diceA.y + (1.0 + FILTER_WIDTH) * linearstep(0.5, 0.0, p7)
      );
      amp *= smoothstep(
        0.0 - FILTER_WIDTH,
        0.0,
        diceA.y - (1.0 + FILTER_WIDTH) * linearstep(0.5, 1.0, p7)
      );

      float pm = 0.2 * smoothstep(0.0, 0.5, p7) * smoothstep(1.0, 0.5, p7);
      phase += pm * fract(4.0 * phase); // add high freq

      sum += amp * env * sin(TAU * phase) / 1000.0;
    }

    dest += sum;
  }

  return clip(1.3 * tanh(dest));
}
