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

#define S2T (15.0 / bpm)
#define B2T (60.0 / bpm)
#define ZERO min(0, int(bpm))
#define saturate(x) clamp(x, 0., 1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define clip(x) clamp(x, -1., 1.)
#define lofi(i,m) (floor((i)/(m))*(m))
#define repeat(i, n) for (int i = ZERO; i < n; i++)
#define p2f(i) (exp2(((i)-69.)/12.)*440.)

const float PI = acos(-1.0);
const float TAU = PI * 2.0;

uniform vec4 param_knob0; // bass amp
uniform vec4 param_knob4; // bass macro
uniform vec4 param_knob5; // oidos filter

#define p0 paramFetch(param_knob0)
#define p4 paramFetch(param_knob4)
#define p5 paramFetch(param_knob5)

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

vec2 rect(float phase, vec2 size) {
  float t1 = 0.5 * size.x / (size.x + size.y);
  vec2 v = vec2(-0.5);

  phase = fract(phase);
  v.x += linearstep(0.0, t1, phase);
  v.y += linearstep(t1, 0.5, phase);
  phase -= 0.5;
  v.x -= linearstep(0.0, t1, phase);
  v.y -= linearstep(t1, 0.5, phase);

  return size * v;
}

vec2 obsvr(float phase) {
  vec2 dest = vec2(0.0);

  float phaset = 9.0 * fract(phase);

  if (phaset < 4.0) {
    dest += rect(fract(phaset / 4.0 - 0.0625), vec2(0.6, 1.0));
  } else if (phaset < 8.0) {
    dest += rect(fract(phaset / 4.0 + 0.0625), vec2(1.0, 0.6));
  } else {
    dest += rect(fract(phaset), vec2(0.2, 0.2));
  }

  dest *= rotate2D(TAU * (0.125));

  return dest;
}

mat3 orthBas(vec3 z) {
  z = normalize(z);
  vec3 x = normalize(cross(vec3(0, 1, 0), z));
  vec3 y = cross(z, x);
  return mat3(x, y, z);
}

vec3 cyclic(vec3 p, float pers, float lacu) {
  vec4 sum = vec4(0);
  mat3 rot = orthBas(vec3(2, -3, 1));

  repeat(i, 5) {
    p *= rot;
    p += sin(p.zxy);
    sum += vec4(cross(cos(p), sin(p.yzx)), 1);
    sum /= pers;
    p *= lacu;
  }

  return sum.xyz / sum.w;
}

vec2 mainAudio(vec4 time) {
  vec2 dest = vec2(0);

  { // 0b5vr
    float l = timeLength.z;
    float t = time.z;

    float freq = 40.0;
    freq = floor(freq * l) / l;

    float macro = exp2(9.0 * (-1.0 + p4));

    const int N_UNISON = 64;

    vec2 sum = vec2(0.0);
    repeat(i, N_UNISON) {
      vec3 dice = hash3f(vec3(i, 22, 89));

      float phase = fract(freq * t);
      phase += 0.5 * macro * sin(TAU * (4.0 * t / l + dice.y));
      sum += obsvr(phase) * rotate2D(macro * TAU * (dice.z - 0.5));
    }

    dest += p0 * exp2(3.0 * macro) * sum / float(N_UNISON);
  }

  { // oidos drone
    vec2 sum = vec2(0.0);

    repeat(i, 2500) {
      vec3 diceA = hash3f(vec3(i / 50, 18, 46));
      vec3 diceB = hash3f(vec3(i));

      float t = mod(time.z - diceA.x * (64.0 * B2T), 64.0 * B2T);
      float env = sin(PI * t / (64.0 * B2T));

      float tone = 6.0 + 7.0 * diceA.y + 0.05 * diceB.y;
      float freq = exp2(tone);
      vec2 phase = t * freq + fract(diceB.xy * 999.0);

      float amp = 1.0;
      const float FILTER_WIDTH = 0.3;
      amp *= smoothstep(
        1.0 + FILTER_WIDTH,
        1.0,
        diceA.y + (1.0 + FILTER_WIDTH) * linearstep(0.5, 0.0, p5)
      );
      amp *= smoothstep(
        0.0 - FILTER_WIDTH,
        0.0,
        diceA.y - (1.0 + FILTER_WIDTH) * linearstep(0.5, 1.0, p5)
      );

      float pm = 0.2 * smoothstep(0.0, 0.5, p5) * smoothstep(1.0, 0.5, p5);
      phase += pm * fract(4.0 * phase); // add high freq

      sum += amp * env * sin(TAU * phase) / 1000.0;
    }

    dest += sum;
  }

  return clip(1.3 * tanh(dest));
}
