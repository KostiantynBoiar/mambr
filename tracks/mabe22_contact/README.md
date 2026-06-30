# tracks/mabe22_contact — PRIMARY track

**The new idea / headline contribution.** Does **inter-animal mask-contact geometry**
(mask IoU/overlap, contact-boundary length) add signal for social/contact behaviours
(Chase, Huddle, Oral Contact, Oral–Genital Contact) **over the keypoint-hull geometry the
field already uses** (SimBA/MARS)? Dataset: **MABe22 mouse-triplet** (multi-animal).

- **Plan:** `ROADMAP.md` (M0–M9, verify-first; **M2 masking pilot = week-1 go/no-go**).
- **Master spec:** `../../.claude/CLAUDE.md`.
- **Make-or-break:** does SAM 2 keep 3 identical mice identity-stable *through contact*?
- **Status:** scaffold only. Code is built one stage at a time.

```
mabe22/        package scaffold: data, masking, geometry, models, scripts, utils, configs
report/        proposal/ + presentation/ (this track's own docs — written later)
ROADMAP.md     the M0–M9 stages
```

Reuses the MammAlps track's patterns (frozen encoders, `Config`, summation-fusion head,
eval-canary, remotezip fetch) — via direct import or `common/` once shared code is extracted.
Build the reports with `report/build.sh` (Tectonic + Ghostscript-normalise for GitHub).
