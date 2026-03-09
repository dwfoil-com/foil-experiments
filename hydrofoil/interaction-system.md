# Hydrofoil Interaction System

This document defines the shared interaction system for the entire `hydrofoil` article. The goal is to make every chapter feel like part of one instrument panel, not a collection of unrelated widgets.

The benchmark is the teaching structure of <https://ciechanow.ski/airfoil/>:

- interactive-first
- visually sparse
- one idea at a time
- scientifically legible
- appealing because it is clear, not because it is overloaded

## Core Principle

Every interactive scene should be built from the same small set of reusable parts:

- one main stage
- one focused question
- one or two controls
- one or two highlighted observations
- one concise takeaway

The user should learn the interaction language once, then reuse it everywhere.

## Shared Scene Anatomy

Each chapter scene should follow this layout:

1. **Question**
   A short title framed as a concrete question.
   Example: "What changes when the rider moves back?"
2. **Stage**
   The main visual area. Usually a side-view hydrofoil scene.
3. **Controls**
   One primary slider or toggle group, sometimes one secondary control.
4. **Readouts**
   A small strip of 3-5 core values.
5. **Observation prompt**
   A sentence telling the reader what to look for.
6. **Takeaway**
   One concise conclusion after the interaction.

This should be the default pattern for almost every section.

## Main Stage System

The main stage is the reusable visual canvas across the article.

### Base view

Default to a clean 2D side view:

- water surface
- board
- mast
- front wing
- tail
- rider body simplified into torso + legs + foot markers

This view is the backbone of the article because it makes trim, pitch, ride height, stance, and structural leverage easy to see.

### Overlays

Only turn on the overlays needed for the current lesson:

- flow arrows
- lift vectors
- drag vectors
- weight vector
- center of mass marker
- center of lift marker
- pitch moment arc
- mast-base reaction forces
- path / motion trail
- pressure or loading heat overlay where justified

The mistake to avoid is showing all overlays at once.

### Camera rules

- Default: fixed side view
- Optional: zoomed detail inset for wing section, mast base, or tail
- Avoid free camera unless a chapter truly needs it

The reader should always feel spatially oriented.

## Reusable Interaction Primitives

These are the core interactions to reuse across chapters.

### 1. Scrub

Used for:

- time
- animation phase
- one pump cycle
- before/after comparison

Behavior:

- drag horizontally to move through time or phase
- play/pause optional
- should feel smooth and reversible

### 2. Single physical slider

Used for:

- rider stance
- speed
- angle of attack
- tail incidence / shim
- ride height
- mast length

Behavior:

- one slider changes one causal variable
- visual response should be immediate
- should snap to notable presets when useful

### 3. Toggle comparison

Used for:

- big tail vs small tail
- balanced trim vs nose-heavy trim
- pump foil vs HA foil
- powered stance vs glide stance

Behavior:

- quick A/B switch
- preserve camera and scene state
- animate the delta instead of hard swapping when possible

### 4. Guided drag target

Used for:

- dragging rider stance along the board
- dragging the center of mass
- moving mast position relative to board

Behavior:

- direct manipulation should only be used when it teaches better than a slider
- the drag target should be obvious and forgiving

### 5. Overlay reveal

Used for:

- showing hidden force vectors
- revealing pressure map
- exposing structural load paths

Behavior:

- default hidden until needed
- smooth fade in/out
- often tied to a "show forces" toggle

## Control System Rules

The control system should stay narrow.

### Default control budget

Per scene:

- 1 primary control
- 0-1 secondary control
- 0-2 toggles

If a scene needs more than that, it is probably trying to teach too much at once.

### Progressive disclosure

Early chapters:

- only expose the one variable under discussion

Later chapters:

- allow limited combinations once the reader understands the building blocks

### Presets over raw freedom

Prefer meaningful presets such as:

- Beginner too far forward
- Neutral trim
- Back-foot sweet spot
- Big forgiving tail
- Small fast tail
- Pump foil
- High-aspect foil

Presets teach better than a wall of numeric controls.

## Visual Grammar

This is the reusable diagram language.

### Color roles

Colors should carry meaning consistently:

- Blue / cyan: water flow and hydrodynamic context
- Green: lift / upward support
- Red / orange: drag / loss / overload
- Yellow: center markers / highlighted geometry
- Gray / charcoal: static structure and baseline geometry

Avoid decorative gradients that do not encode anything.

### Vector style

- Solid arrows for real forces
- Dashed arrows for velocity or hypothetical motion
- Curved arc for pitch moment
- Thicker arrows for dominant forces
- Animated pulse only when drawing attention to a change

### Marker style

- Center of mass: filled circle
- Center of lift: ring or crosshair
- Contact / reaction points: square markers
- Motion trail: thin fading path

### Labels

- Put labels near the thing, not in a legend when possible
- Keep text short
- Use the same nouns throughout the article

## Readout Strip

Every scene can optionally use the same compact readout strip.

Good readout candidates:

- speed
- pitch
- ride height
- front-wing lift
- tail lift
- total drag
- net pitch moment
- front-box compression
- rear-box tension

Rules:

- show only the values relevant to the scene
- maximum 5 values visible
- emphasize change, not dashboard density

## Motion and Transitions

Motion should explain causality, not decorate the page.

### Good uses of motion

- interpolate force-vector changes as a slider moves
- animate the rider shifting stance
- scrub through one pump cycle
- show force-path changes when overlays appear

### Avoid

- floaty UI animations
- constant idle motion with no teaching value
- dramatic transitions that break continuity

## Recommended Scene Templates

### Template A: Causal Slider

Best for:

- stance
- speed
- ride height

Structure:

- side-view stage
- one slider
- 3 readouts
- one observation prompt

### Template B: A/B Comparison

Best for:

- tail size comparison
- trim comparison
- foil family comparison

Structure:

- same scene, same camera
- toggle between A and B
- delta overlay shows what changed

### Template C: Pump Cycle Scrubber

Best for:

- pumping mechanics
- load spikes
- timing explanation

Structure:

- one pump phase scrubber
- force vectors update with phase
- optional energy / speed mini-chart below

### Template D: Overlay Reveal

Best for:

- structural loads
- pressure intuition
- hidden balance forces

Structure:

- start simple
- click or toggle to reveal invisible quantities
- short explanation follows

## Cross-Chapter Persistent Concepts

These concepts should look identical everywhere:

- rider center of mass
- front-wing lift
- tail lift
- drag
- pitch moment
- stance position
- mast-base reactions

If these visual encodings drift chapter to chapter, the article will feel harder than it needs to be.

## Interaction Map By Topic

### Trim

Primary interaction:

- drag rider stance or use stance slider

Secondary:

- show/hide force vectors

### Tail / Shim

Primary interaction:

- shim slider or tail preset toggle

Secondary:

- stance lock versus balance lock

### Pumping

Primary interaction:

- pump phase scrubber

Secondary:

- stance preset

### Surface Effects

Primary interaction:

- ride-height slider

Secondary:

- speed preset

### Structure

Primary interaction:

- front-foot / back-foot load bias slider

Secondary:

- mast length or plate-size preset

## Implementation Guidance

This interaction system suggests a component model roughly like:

- `SceneFrame`
- `HydrofoilStage`
- `ControlBar`
- `ReadoutStrip`
- `ObservationCallout`
- `TakeawayNote`
- reusable overlay layers for forces, centers, moments, and structure

The same scene shell should wrap all chapter-specific simulations.

## Quality Bar

A scene is good enough when:

- the reader understands what changed without reading every label
- the main variable feels directly connected to the visual response
- the scene stays legible when paused on any frame
- the controls are obvious within a few seconds
- the scene teaches one thing clearly

A scene is not good enough when:

- it looks like engineering software
- it needs a long paragraph to decode
- too many values are moving at once
- the reader can play with it without learning the intended point
