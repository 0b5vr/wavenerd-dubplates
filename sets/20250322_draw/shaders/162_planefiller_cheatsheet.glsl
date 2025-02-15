/*
planefiller cheatsheet

- make sure knob4 is zero
- 1.01
  - introduce the track
- 1.09
  - ➕️ kick
  - ➕️ rim
- 1.16
  - 🎛️ low cut
- 2.01
  - ➕️ hihat
  - ➕️ fm perc
  - ➕️ crash
  - ➕️ bass
  - 🎛️ low undo
- 2.09
  - ➕️ clap
  - 🏗️ kick fillptn 0x88a6
- 3.01
  - ➕️ open hihat
- 3.09
  - 🎛️ low cut
  - 🎛️ gradually increase choir length
  - ➕️ crash pattern change
  - 🏗️ kick fillptn 0x808f
- 3.16
  - 🎛️ low undo
- 4.01
  - ➕️ ride
  - ➕️ arp
- 4.09
  - ❌️ snare909 longer roll
  - 🏗️ kick fillptn 0x809e
- 5.01
  - 🏗️ Transpose by -3
- 5.09
  - 🏗️ kick fillptn 0x808f
- 6.01
  - ❌️ hihat
  - ❌️ open hihat
  - ❌️ fm perc
  - ❌️ rim
  - ❌️ ride
  - ❌️ clap
  - ❌️ snare909
*/

vec2 mainAudio(vec4 time) {
  return vec2(0.0);
}
