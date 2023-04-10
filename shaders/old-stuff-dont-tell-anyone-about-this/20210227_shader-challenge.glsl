#define BPM bpm
#define PI 3.14159265
#define beat *(60.0/BPM)
#define TAU (PI*2.)
#define lofi(i,j) ( floor( (i)/(j) ) * (j) )

float fs(float s){
  return fract(sin(s*114.514)*1919.810);
}

float func(float x, float y, float z) {
    z = fract(z), x /= pow(2.,z), x += z*y;
    float v = 0.;
    for(int i=0;i<6;i++) {
        v += asin(sin(x)) * (1.-cos((float(i)+z)*1.0472));
        v /= 2., x /= 2., x += y;
    }
    return v * pow(2.,z);
}

vec2 mainAudio(vec4 wntime){
  float time=wntime.z;
  vec2 ret=vec2(0);
  
  // glitch
  if(mod(time, 64.0 beat)>60.0 beat){
    if(mod(time - 0.0 beat,4.0 beat)<1.0 beat){
      float retrigger=0.5 beat;
      time-=lofi(mod(time,1.0 beat),retrigger);
    }

    if(mod(time - 1.0 beat,4.0 beat)<1.0 beat){
      float retrigger=(mod(time, 1.0 beat)<0.5 beat) ? (0.0625 beat) : (0.25 beat);
      time-=lofi(mod(time,1.0 beat),retrigger);
    }

    if(mod(time - 2.0 beat,4.0 beat)<1.0 beat){
      time=lofi(time,0.0001+0.001*mod(lofi(time, 0.01), 1.0 beat));
    }

    if(mod(time - 3.0 beat,4.0 beat)<1.0 beat){
      time-=0.8*lofi(mod(time,1.0 beat),.04);
    }
  }
  
  float sidechain=1.;
  
  { // kick
    float t=mod(time,1.0 beat);
    sidechain=mix(.2,1.,smoothstep(.0 beat, .6 beat, t));
    float amp=6.0*exp(-3.*t);
    float env=exp(-3.*t);
    float phase=220.0*t;
    float p=1000.0*exp(-170.*t)+500.0*exp(-20.*t)-phase*TAU;
    vec2 wave=vec2(
      func(p,166.0*env-t,30.0*t),
      func(p,166.0*env+t,30.0*t)
    );
    ret+=.5*clamp(amp*wave,-1.,1.);
  }
  
  { // hat
    float s=lofi(time, 0.25 beat);
    float t=time-s;
    float amp=exp(-mix(10.,40.,fs(s))*t);
    float phase=800.0+mix(120.,200.,fs(s+1.))*t;
    vec2 wave=vec2(
      func(4.0*phase*TAU,28.0,0.5*sin(phase*TAU)),
      func(4.0*phase*TAU,29.0,0.5*sin(phase*TAU))
    );
    ret+=sidechain*.3*clamp(amp*wave,-1.,1.);
  }
  
  { // shaker
    float s=lofi(time, 0.25 beat);
    float t=lofi(time-s,.00015);
    float amp=exp(-mix(10.,40.,fs(s+1.77))*t);
    float phase=800.0+mix(200.,200.,fs(s+2.74))*t;
    vec2 wave=vec2(
      func(4.0*phase*TAU,28.0,0.2*sin(phase*TAU)-1.*lofi(time,.02)),
      func(4.0*phase*TAU,29.0,0.2*sin(phase*TAU)-1.*lofi(time,.02))
    );
    ret+=sidechain*.3*clamp(amp*wave,-1.,1.);
  }
  
  { // saw?
    float s=lofi(time, 0.25 beat);
    float t=time-s;
    float amp=5.*exp(-mix(10.,100.,fs(s+4.56))*t);
    float phase=800.0+mix(10.,30.,fs(s+1.17))*t;
    vec2 wave=vec2(
      func(phase*TAU,0.,10.0*sin(.07*phase*TAU)),
      func(phase*TAU,1.,10.0*sin(.07*phase*TAU))
    );
    ret+=sidechain*.1*clamp(amp*wave,-1.,1.);
  }
  
  { // clap
    float t=mod(time - 1.0 beat,2.0 beat);
    float amp=2.*exp(-10.*t);
    float phase=700.0*t;
    vec2 wave=vec2(
      func(3000.0*exp(-20.*t),28.0,1.0*sin(phase*TAU*t)),
      func(3000.0*exp(-20.*t),29.0,1.0*sin(phase*TAU*t))
    );
  	ret+=sidechain*.5*clamp(amp*wave,-1.,1.);
  }
  
  { // ride
    float s=lofi(time, 0.5 beat);
    float t=time-s;
    float amp=exp(-5.*t);
    float phase=800.0+800.0*t;
    vec2 wave=vec2(
      func(4.0*phase*TAU,11.0,0.5*sin(phase*TAU)),
      func(4.0*phase*TAU,12.0,0.5*sin(phase*TAU))
    );
    ret+=sidechain*.2*clamp(amp*wave,-1.,1.);
  }
  
  { // beep
    float t=mod(time - 0.25 beat,2.0 beat);
    float amp=2.*exp(-10.*t);
    float phase=480.0*t;
    vec2 wave=vec2(
      func(400.0*exp(-20.*t),28.0,1.0*sin(phase*TAU)),
      func(400.0*exp(-20.*t),29.0,1.0*sin(phase*TAU))
    );
    ret+=sidechain*.3*clamp(amp*wave,-1.,1.);
  }

  return ret;
}
