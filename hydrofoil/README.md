# Hydrofoil

Build a hydrofoil-focused interactive explainer in the spirit of <https://ciechanow.ski/airfoil/>, but aimed at surf hydrofoils with a rider on board. The target is a comprehensive, visual, simulation-backed piece that explains not just foil section aerodynamics, but the full rider-foil-board system: trim, stance, pumping, turning, tail/stab tradeoffs, ventilation, breach, and structural loads into the mast tracks and board.

## Summary

The core idea is to treat a surf hydrofoil as a coupled system:

- Water flow around front wing, mast, fuselage, and tail
- Rider center of mass and foot placement relative to lift
- Board pitch, heave, and speed
- Free-surface effects near breach or touchdown
- Load transfer into the mast, tracks, bolts, and board structure

This is broader than a wing-section explainer. It needs to answer the questions riders actually ask when foils feel "fast", "firm", "mushy", "tracky", "turny", "easy to pump", or "hard to recover".

## Goal

Produce an interactive article and simulation toolkit that makes surf hydrofoil behavior legible to non-engineers without dumbing it down. The benchmark is "similarly comprehensive" to the airfoil article, but grounded in surf hydrofoil scenarios:

- Prone / paddle-up
- Pumping and linking
- Wingfoil transition from powered stance to glide stance
- Foil Drive motor-to-pump transition
- Downwind glide and recovery
- Low-speed turn mode versus high-efficiency glide mode

## Users

- Surf hydrofoil riders trying to understand trim, stance, and pumping
- Shapers and foil designers reasoning about front wing / tail / shim tradeoffs
- Board builders interested in mast-track and deck load cases
- Curious readers who want a better mental model than forum folklore

## Product Shape

The likely output is a long-form interactive web article with embedded simulations and diagrams.

Planned ingredients:

- Scroll-driven explanatory sections
- Interactive free-body diagrams
- Live plots for lift, drag, pitch moment, and stance location
- A simplified rider + foil dynamics view
- Before/after comparisons for trim, tail size, and shim changes
- Short scenario presets using real rider questions

See [interaction-system.md](interaction-system.md) for the shared scene and control framework that should be reused across all chapters.

## Style Benchmark: What To Copy From `airfoil`

This project should imitate the teaching style of the `airfoil` article, not just its topic ambition.

What stands out about <https://ciechanow.ski/airfoil/>:

- It is interactive from the beginning, not only at the end.
- It teaches with experiments, not just illustrations.
- It starts from simple physical primitives and only later assembles the full phenomenon.
- It uses one variable at a time, with controls that are narrow and meaningful.
- It keeps a consistent visual grammar across the whole piece.
- It is scientific without reading like a textbook.
- It is interesting because each section creates a small "oh, that is why" moment.

That means `hydrofoil` should avoid the common failure mode of interactive explainers:

- giant wall of prose
- one fancy but shallow demo
- too many controls at once
- unexplained jargon
- realism that is too complex to teach with

### Style Principles For `hydrofoil`

1. Start concrete, not abstract.
   Open with a rider-on-foil scenario people recognize, then back out into the underlying physics.
2. Build intuition in layers.
   Introduce flow, lift, trim, tail balance, pumping, and structural loads as separate pieces before combining them.
3. Make each interactive scene answer one question.
   Every demo needs a crisp teaching objective, not just "play around with a simulation".
4. Use a stable visual language.
   Reuse the same arrows, trails, pressure colors, moment arms, and force vectors throughout the article.
5. Keep controls sparse.
   Time, speed, stance, angle of attack, shim, and tail size are enough for early scenes. Hide everything else until needed.
6. Show the invisible.
   Pressure, center of lift, rider center of mass, pitch moment, and board reaction loads should all become directly visible.
7. Admit simplifications explicitly.
   If a scene is 2D, quasi-steady, or ignoring roll/yaw/waves, say so clearly.
8. Keep it interesting with surprise.
   The best moments will likely be counterintuitive ones: moving back reduces drag, bigger tails can improve "low end" while hurting efficiency, and front-foot pumps can punish the board more than riders expect.

### Practical Implications

For implementation, each chapter should probably have:

- one core interactive
- one or two carefully chosen controls
- one short setup paragraph
- one or two observations the reader should notice
- one concise takeaway before moving on

That structure is much closer to `airfoil` than a conventional article with occasional diagrams.

## Proposed Chapter Outline

1. What a surf hydrofoil system actually is
2. Lift, drag, angle of attack, and why water changes the feel
3. Where the rider stands: center of mass, center of lift, and trim
4. Why moving back can make a foil feel faster, firmer, and easier to pump
5. Front wing versus tail: who is lifting what, and at what drag cost
6. Shim, tail incidence, and the difference between balance and compensation
7. Pumping mechanics: "jump up" versus "push down"
8. Turn mode versus efficiency mode at different speeds
9. Breach, ventilation, cavitation, and surface interaction
10. Structural loads: mast, bolts, tracks, deck compression, rear tension
11. What changes across foil families: big pump foils, mid-aspect surf foils, high-aspect / race-like foils

See [content-outline.md](content-outline.md) for a more detailed section-by-section breakdown.

## Physics Approach

The sensible default is to use `foilphysics` as the starting point for the force model, then layer surf-hydrofoil-specific state and interactions on top.

Primary references:

- `airfoil` benchmark article: <https://ciechanow.ski/airfoil/>
- `foilphysics` model: <https://github.com/lsegessemann/foilphysics>
- `dynamicsim` implementation/example: <https://foilien.com/foilphysics/dynamicsim/dynamicsim>

### Why `foilphysics`

It already provides a usable base for foil force calculations and dynamic simulation. That should be good enough for the bulk model, especially for:

- Lift and drag versus angle of attack
- Time-stepped motion and stability experiments
- Comparing setup changes under controlled inputs

### What needs to be added for this project

`foilphysics` is not the full product. This project needs extra layers that matter specifically for surf hydrofoils:

- Rider stance as a movable input, not a fixed load
- Front foot / back foot weighting and stance transitions
- Board + mast + fuselage geometry as a lever system
- Surface-piercing / near-surface behavior
- Pump-cycle inputs instead of steady cruise only
- "Turn mode" versus "efficiency mode" presets
- Structural load calculations into boxes and bolts

## Core Research Questions

The first pass should answer the misunderstandings that show up repeatedly in rider discussions:

- Why does moving the front foot back often make a foil feel more balanced and more efficient?
- When does a big tail help low end, and when is it just extra drag?
- What is the actual difference between "can go slow" and "can be recovered when going slow"?
- Why do some foils feel "firm" and others feel "mushy" at the same rider weight?
- Why do riders step back to pump, but sometimes step forward again when speed gets low?
- How much of "more shim" is genuine optimization versus compensation for poor trim?
- What does the tail stabilize, and what would happen without it?
- Which forces dominate mast-track and board stress during pumping?

See [misconceptions.md](misconceptions.md) and [forces-and-structure.md](forces-and-structure.md).

## Simulation Requirements

Minimum useful simulation state:

- Forward speed
- Pitch angle and pitch rate
- Heave / ride height
- Wing and tail angle of attack
- Rider center of mass relative to mast
- Front foot / back foot load split
- Front-wing lift and drag
- Tail lift and drag
- Resulting pitching moment
- Approximate mast-base and bolt loads

Good first simplifications:

- 2D side-view model before full 3D
- Lump rider + board mass, but keep stance as a separate lever input
- Quasi-steady foil coefficients before chasing transient CFD-style effects
- Preset foil families rather than exact brand-accurate geometry on day one

## Deliverables

- Long-form article / site architecture
- Reusable hydrofoil simulation model
- Scenario presets for common rider questions
- Visual diagrams for trim, pumping, and structural loads
- Notes translating forum language into testable physics claims

## Initial Roadmap

1. Build the content architecture and misconception list.
2. Stand up a minimal dynamics sandbox around `foilphysics`.
3. Add rider stance and foot-pressure inputs.
4. Validate against observed foil behaviors in simple scenarios.
5. Create the article scenes and diagrams.
6. Add structural load views for mast tracks and board construction.

## Notes

This project should stay honest about what is explanatory physics versus what is rider heuristic. A lot of foil advice is directionally right but mixes steady-state trim, transient pumping, and subjective feel. The article should separate those cleanly.
