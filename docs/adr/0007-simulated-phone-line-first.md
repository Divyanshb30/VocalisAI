# ADR 0007: Faithful phone-line simulation before real telephony

**Status:** accepted

## Context
Real calls cost money, are hard to reproduce, and must not target real businesses.

## Decision
Simulate the line: 8 kHz μ-law round trip, band-pass, noise, in-band DTMF generated as audio and decoded with Goertzel, IVR menus and hold. Real Twilio red-team calls come later on the same pipeline.

## Alternatives considered
Twilio from day one.

## Consequences
Reproducible evals at $0; Twilio media streams cannot send DTMF anyway, so in-band tones are the real path.
