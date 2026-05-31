//                                                                                   
//    ▀██                            ▄█▀                             ▀█▄             
//   ▄▄██  ▄▄ ▄▄▄   ▄▄▄▄   ▄▄   ▄▄  ██  ▄▄▄▄▄ ▄▄▄▄▄ ▄   ▄ ▄   ▄ ▄▄▄▄▄  ██  ██        
// ▄█▀ ██   ██▀ ██  ▄▄▄██  ██ ▄ ██  ██    █   █   █ █▄▄▀   ▀▄▀  █   █  ██            
// ██  ██   ██     ██  ██  ██ █ ██  ██    █   █▄▄▄█ █  ▀▄   █   █▄▄▄█  ██  ▄▄        
// ▀█▄▄▀█▄ ▄██▄    ▀█▄▄▀█▄ ▀██▀██▀   ▀█▄                             ▄█▀  ▄█▀        
//                                                                                   
// Audio/Visual Event "draw(tokyo); #4"                                              
// CIRCUS TOKYO @ Shibuya, Tokyo                           2026-05-30 14:00 - 20:30  
//                                                                                   
//                           0b5vr x gam0022 x kinankomoti                           

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define TRANSPOSE 0.0

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

vec2 cis(float t) {
  return vec2(cos(t), sin(t));
}

mat2 rotate2D(float t) {
  float c = cos(t);
  float s = sin(t);
  return mat2(c, s, -s, c);
}

float tmod(vec4 time, float d) {
  vec4 t = mod(time, timeLength);
  float offset = lofi(t.z - t.x + timeLength.x / 2.0, timeLength.x);
  offset -= lofi(t.z, d);
  return t.x + offset;
}

const int N_CHORD_NOTES = 8;
int CHORD[] = int[](
  5, 12, 12+2, 12+5, 12+7, 24, 24+2, 24+7,
  4, 12, 12+4, 12+7, 12+11, 24, 24+2, 24+7
);

vec2 mainAudioDry(vec4 time) {
  vec2 dest = vec2(0);

  { // plucks
    vec2 sum = vec2(0.0);
    repeat(i, 64) {
      vec3 dice = hash3f(vec3(i / 4, 12, 18));
      vec3 dicetime = hash3f(vec3(i / 4, 111, 71));

      float delay = float(i % 4);
      float delaydecay = exp2(-delay);

      vec4 tdelay = mod(time - delay * 4.0 * S2T, timeLength);
      float l = 2.0 * S2T * floor(3.0 + 9.0 * dicetime.x);
      float t = tmod(tdelay + 2.0 * S2T * floor(9.0 * dicetime.y), l);
      float q = l - t;
      float st = round((tdelay.w - t) / S2T);

      float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, q);
      env *= smoothstep(0.0, 0.01, t);
      env *= exp2(-10.0 * t);

      int iNote = int(dice.x * float(N_CHORD_NOTES) * 2.5) % N_CHORD_NOTES;
      iNote += mod(st, 64.0) < 32.0 ? 0 : N_CHORD_NOTES;
      float pitch = 48.0 + TRANSPOSE + float(CHORD[iNote]);
      pitch += 12.0 * floor(2.0 * dice.y);
      float freq = p2f(pitch);
      float phase = t * freq + dice.y;
      phase = lofi(phase, 1.0 / 64.0);

      float fm = 0.4 * dice.y * sin(TAU * phase);
      vec2 wave = vec2(sin(TAU * phase + fm));

      sum += delaydecay * env * wave * rotate2D(TAU * float(dice.z));
    }

    dest += 0.12 * sum;
  }

  return dest;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = mainAudioDry(time);
  return dest;
}
