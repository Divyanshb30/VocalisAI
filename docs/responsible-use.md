# Responsible use

VocalisAI makes phone calls on a person's behalf. These rules are enforced in code where possible and documented here where they are policy.

## Always
- **Disclose.** The first thing the agent says on every call is that it is an AI assistant calling on behalf of a named passenger. It never denies being an AI when asked.
- **Stay inside the mandate.** The passenger approves the target outcome, acceptable fallbacks and a walk-away point before the call. Any offer outside the mandate is referred back to the passenger.
- **Hand off for sensitive steps.** OTPs, payments, identity verification and anything the passenger did not pre-authorise are handed to the passenger live.
- **Minimise data.** OTPs, full card numbers, CVVs, passport numbers and passwords are never stored and are redacted when documents are read. Permissioned data (phone, date of birth, last four card digits) is only spoken if the passenger allowed it for that case.

## Never
- Call real businesses in this version. v1 runs only against the SimAir simulator and consenting human testers.
- Call emergency numbers, place marketing or bulk calls, or run more than one call at a time per passenger.
- Record a call without announcing it; if the other party objects, recording stops.
- Use real personal documents in the public demo. The demo uses synthetic documents only, because free-tier model providers may retain inputs.

## Legal note
Passenger-rights figures are computed from the published regulations (DGCA CAR Section 3 Series M Part IV, UK261/EU261, UAE GCAA guidance) and cite their source clause. They are informational, not legal advice.
