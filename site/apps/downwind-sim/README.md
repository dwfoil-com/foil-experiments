# Downwind Foiling Wave Sim — built bundle

This directory holds a **built** copy of a downwind foiling simulator, served at
`/foil-experiments/apps/downwind-sim/`. Do not edit these files by hand; they
are compiled output.

## Provenance and credit

The original game was written by **gecko39 / cclaan** and lives at
<https://github.com/cclaan/downwind-sim>, with the author's own deployment at
<https://usedwatersports.com/downwind-sim>. The in-game "View code on GitHub"
banner links back to that repository.

> **The upstream repository carries no licence file.** Absent a licence, default
> copyright applies and there is no explicit grant to redistribute or host a
> modified copy. The author has publicly invited suggestions and said they may
> build a v2, but that is not the same as permission to rehost. Get the author's
> agreement before treating this as a permanent part of the site, or point the
> link at their deployment instead.

## What this build changes

Work sits on the `feat/wave-trains` branch of the fork at
<https://github.com/mattarderne/downwind-sim>:

- Wave field rebuilt on deep-water linear theory. Swells are specified as
  significant height, peak period and direction; wavelength, crest speed and
  group speed follow from `omega^2 = g*k`.
- Two swell systems plus wind chop, each expanded into a band of spectral
  components so wave groups (sets) emerge from beating rather than being
  scripted. Group speed measures at roughly half crest speed, as theory
  requires, so crests march forward through the set.
- Foil flight driven by angle of attack rather than speed alone, with orbital
  velocity feeding the inflow, free-surface lift loss near the surface, and a
  breach failure mode.
- Instrument cluster: wave train profile, set position, swell radar, and a trim
  and height gauge drawn as a side elevation of the rig against the water.
- Sea state config panel with small/medium/large presets per swell.

## Rebuilding

From a checkout of the fork:

```bash
npm ci
npx vite build \
  --base=/foil-experiments/apps/downwind-sim/ \
  --outDir /path/to/foil-experiments/site/apps/downwind-sim \
  --emptyOutDir
rm -f /path/to/foil-experiments/site/apps/downwind-sim/paddle.fbx  # unused asset
```

The `--base` override matters: the fork's `vite.config.ts` targets the author's
own host path.

## Known limitations here

- **The global leaderboard is inert.** It is backed by an API that only exists
  on the author's host, so the build detects the missing endpoint and shows
  "Offline preview — scores are not ranked". Splits and total time still work.
- Upstream's analytics tag is stripped, so this build reports nothing.
