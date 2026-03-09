# Style Analysis: `airfoil`

Source analyzed: <https://ciechanow.ski/airfoil/>

## High-Level Takeaway

The article works because it combines three things at once:

- scientific rigor
- visual clarity
- a steady sequence of interactive experiments

It does not dump theory first and add widgets later. The interactions are the explanation.

## What The Article Actually Does

### 1. It starts with tangible analogies

Early sections use grass, leaves, arrows, markers, particles, tennis balls, and simple boxes before asking the reader to reason about a full airfoil. That reduces abstraction cost.

Implication for `hydrofoil`:

- Start with a rider, board, mast, and simple force arrows
- Use concrete surf scenarios before abstract coefficient plots

### 2. It introduces one concept at a time

The article separates:

- flow visualization
- velocity
- relative velocity
- pressure
- viscosity
- boundary layer behavior
- only then the airfoil itself

Implication for `hydrofoil`:

- Do not start with the entire foil system UI
- Separate flow, lift, trim, tail balance, pumping, and structure into staged scenes

### 3. It makes the invisible visible

The piece repeatedly turns hidden quantities into visible ones:

- arrows for local velocity
- markers and trails for pathlines
- color maps for speed
- highlighted collisions for pressure
- overlaid profiles and gradients

Implication for `hydrofoil`:

- Show lift vectors, drag vectors, pitch moment, center of mass, center of lift, tail effort, and mast-base reactions directly

### 4. It uses small, meaningful controls

The controls tend to be:

- time
- one physical parameter
- sometimes camera / viewpoint

This keeps the reader focused on causality.

Implication for `hydrofoil`:

- Each scene should expose only the one or two parameters needed to make the lesson legible
- Avoid a giant "foil configurator" as the main teaching surface

### 5. It is rigorous but still conversational

The tone is careful, explanatory, and technically grounded, but it stays readable through:

- short paragraphs
- plain-language transitions
- concrete observations
- modest use of analogy

Implication for `hydrofoil`:

- Keep the writing clean and specific
- Use rider language where helpful, then translate it into physics

### 6. It is honest about simplification

The article explicitly says when it is using a 2D view or a simplified setup. That earns trust.

Implication for `hydrofoil`:

- Label 2D assumptions, omitted roll/yaw, omitted wave coupling, and any quasi-steady approximations

### 7. It creates curiosity through surprise

Many scenes reveal something non-obvious once the user touches the control.

Implication for `hydrofoil`:

The strongest scenes will likely be ones where intuition is commonly wrong:

- moving the rider back can reduce drag
- tail lift can be stabilizing but costly
- pumping from too far forward can waste effort
- "low end" is not a single variable
- board feel can degrade from flex before obvious failure

## Style Rules For This Project

1. Every interactive must answer a single question.
2. Every quantity the reader needs should be visualized directly.
3. Every chapter should escalate from intuitive to formal.
4. Every simplification should be disclosed.
5. Every control should earn its place.
6. The article should feel exploratory, not encyclopedic.

## Anti-Patterns To Avoid

- Starting with coefficient plots and jargon
- Overloaded scenes with too many sliders
- Treating the rider as a fixed point mass with no stance logic
- Hiding the causal chain between stance, tail load, drag, and feel
- Making the piece look scientific without actually teaching anything
