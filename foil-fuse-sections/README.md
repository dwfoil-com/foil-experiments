# foil-fuse-sections — what the water sees at pump angle

**Status:** poc

Interactive playground for slicing hydrofoil fuselages at the angle they actually fly at, and
looking at the section the water flows over.

Open `playground.html` in a browser. No build, no dependencies, no server.

---

## Credit

The idea comes from [CRISP Foils](https://crispfoils.com) (`@crisp_og`), who posted a low-drag
fuselage concept built around exactly this observation: at the average pump angle the fuse is not
aligned with the flow, so the section presented to the water should be designed as an aerofoil at
that angle rather than left as whatever falls out of extruding a bar.

This project is an independent reconstruction of the geometry from first principles. It is not
CRISP's design, and none of the numbers here come from them.

---

## The idea

Draw a fuselage and you draw a bar. But a fuse only sits parallel to the flow in a picture. In
flight the whole foil is pitched nose-up so the front wing can make lift, and during a pump stroke
that angle swings a long way further. The angle is larger than most people assume.

Once the fuse is at an angle, the streamwise cut through it is not its cross-section. It is a long
thin section whose chord is roughly the fuse height divided by the sine of the angle. At 10&deg;, a
38 mm tall fuse presents a 219 mm chord to the water. That is a real aerofoil, and nobody designed
it.

Worse, its shape is not free. Working through the geometry:

Put the fuse axis along `x` with the nose at `x = 0`, lateral across the fuse as `y`, vertical as
`z`. With the fuse nose-up by `α`, the oncoming water runs along `V(cos α, 0, sin α)`. For a tall
narrow fuse the flow is essentially two-dimensional in the planes that cut across the thin
direction — the same way flow over a mast is two-dimensional in horizontal planes. Those planes are
`−x sin α + z cos α = d`. Writing `u = x cos α + z sin α` for the chordwise coordinate:

```
half-thickness(u) = ( w(x) / 2 ) · Φ(ζ)

  ζ = 2z / h(x)
  x = u cos α − d sin α
  z = u sin α + d cos α
```

`Φ` is the fuse's own cross-section outline: half-width as a fraction of the maximum, plotted
against normalised height. Because `ζ` runs linearly from −1 to +1 along the cut, the result is:

> **The streamwise aerofoil's thickness distribution is the fuse cross-section outline, stretched
> along the chord.**

Square-cornered fuse, and the water sees a rectangle: blunt nose, blunt base, guaranteed
separation. Elliptical fuse, and it sees an ellipse. If you want the water to see a proper
aerofoil, the fuse cross-section has to *be* that aerofoil, stood on end, with the rounded nose
pointing down.

Two consequences worth sitting with:

- **The corner radii on a fuse are aerofoil design, not cosmetics.** They set the nose and tail
  shape of the section the water actually flows over.
- **Which way up matters.** Nose-up means the cut enters through the bottom of the fuse, so an
  asymmetric section only has its rounded nose facing the flow in one direction. A pump stroke
  swings both ways, but not symmetrically.

The model passes through `α = 0` continuously: the cut degenerates to the horizontal plan section
of chord `L`, which is the familiar picture of a fuse.

## What the playground shows

1. The fuse in 3D, cut away at the plane so the section face is exposed. Drag to rotate, scroll to
   zoom, double-click to reset. The near half is drawn as a ghost so the whole shape stays readable.
2. Side view with the flow arriving at the pump angle and the cut marked over its chord.
3. The resulting section drawn to scale, against one baseline of your choosing.
4. Pressure distribution and boundary layer for that section.
5. An optimiser, described below.
6. A pump-cycle model that works out what angle the fuse actually sees.
7. Fuse drag against pump angle for every shape at once, with the pump cycle's range shaded.
8. All six shapes cut at the same angle on a common chord axis, with a table.

## The optimiser

Minimising drag alone is meaningless, because it just drives the fuse to zero width. So the
optimiser holds two things at least as good as the fuse you already have:

- **vertical bending stiffness** `Iyy = w·h³·I₂/8`, which is what resists the tail's downforce
- **cross-sectional area** `A = w·h·I₀/2`, standing in for weight

`I₀` and `I₂` are integrals of the outline function, so they capture shape as well as size. The
stiffness floor sets the thinnest allowable width at any height, `w_min(h) = 8·Iyy*/(h³·I₂)`, and
the area ceiling sets the tallest useful height. Everything between is a valid design, so the
optimiser walks that range for every candidate outline and reports the lowest-drag combination,
subject to a height limit for what the mast and wing joints will take.

**Read the margin, not the winner.** The objective only sees thickness ratio, wetted area and how
blunt the ends are. It separates clean outlines from blunt ones decisively and can barely tell two
clean outlines apart. A plain ellipse currently wins from every starting point, with a designed
aerofoil section within about 1%. That 1% is inside the model's noise and should not be read as a
result. The 93% against a square-edged bar should.

## The pump cycle

The fuse angle relative to the water is `θ − γ`, which is also the front wing's angle of attack. So
the fuse angle is not a free choice. It is whatever the lift equation demands at your speed and
loading:

| Speed | Fuse angle at trim |
|---|---|
| 3.5 m/s | 8.9° |
| 4.5 m/s | 5.4° |
| 7.0 m/s | 2.3° |

Those come from `foil-rl-pump/python-rl/foil_env/foil_physics.py` in this repo, whose
`compute_trim_angle` solves the full wing, stabiliser and mast balance. The playground reimplements
just the wing term and lands within about 2.5% of it.

On top of trim, the pump swings the flight path by `atan(A·ω/V)`, which is large: 80 mm of heave at
2 Hz and 4.5 m/s is ±12.6°. The rider feathers most of that out by pitching with the stroke, and
what is left is the swing the fuse actually works through.

The headline is not that pumping adds angle. It is that **trim itself is 5 to 9° at pumping speeds**,
which is far more than the one or two degrees most people would picture. Slow, high-lift pumping is
exactly where the fuse is most badly aligned.

## Drag estimates

Two independent estimates run side by side, because neither alone deserves much trust.

**Headline number.** Every parallel cut across the body is built, each is given a section drag
coefficient from flat-plate skin friction with a Hoerner strut form factor `1 + 2(t/c) + 60(t/c)⁴`
applied to the real wetted perimeter, plus separate blunt leading-edge and blunt base terms
proportional to the cut-off thickness. Those are integrated across the body. Seawater at
`ν = 1.05×10⁻⁶` m²/s, `ρ = 1025` kg/m³.

**Section solve.** The selected cut goes through a constant-strength source panel method for the
inviscid surface velocity, then a Thwaites laminar boundary layer, Michel transition, Head's
entrainment method for the turbulent part, and Squire–Young for profile drag. This is where
separation shows up.

Both are two-dimensional strip estimates. They ignore the junctions at each end, the tail's
downforce, the front wing's downwash, and all three-dimensional relief. Absolute numbers are
indicative. **Differences between shapes at the same angle are the useful output.**

## What it does not do yet

- **The blunt-end terms are doing most of the work, and they are the least trustworthy part.** On
  the default fuse, 84% of the section drag comes from the blunt leading edge and base terms, which
  are two hand-set constants (0.3 and 0.6 on the cut-off thickness). Worse, a strip model treats
  every parallel cut as an independent 2D section, so the flat top and bottom of the fuse are
  treated as bluff bases. In reality a long flat surface lying at 8° to the flow behaves more like
  a plate at incidence than a base, so the penalty is probably overstated. Treat the direction as
  solid and the magnitude as an upper bound.
- No real CFD. A RANS or even a proper 2D Navier-Stokes solve would be the honest next step,
  and the geometry export needed to feed one does not exist yet.
- The optimiser's objective is too coarse to rank clean outlines against each other. Driving it
  from the panel and boundary layer solve instead of the form-factor estimate would fix that, at
  the cost of being far slower and much more fragile on blunt shapes.
- Torsional stiffness is not constrained, only bending. A tall thin blade gives up lateral
  stiffness quickly, and the optimiser is free to spend it.
- No junction modelling at the mast, front wing, or tail.
- The pump model assumes a sinusoidal heave and a constant feathering fraction. Driving it from a
  trained policy rollout in `foil-rl-pump` would be more honest.
- The bundled shapes are archetypes for exploring the geometry, **not measured brand geometry.**

## Contributing a fuse

The shapes bundled in the playground are my guesses at plausible archetypes. Measured geometry from
a real fuse is far more interesting.

Open the "Add a fuse" panel at the bottom of the playground, set the sliders to match what you have,
and copy the config block. Drop it into `fuses/` as a `.json` file and open a pull request, or paste
it into an issue. Callipers and a brand name beat my guesses every time.

```json
{
  "id": "example-850",
  "name": "Example brand 850",
  "L": 640,
  "h": 38,
  "w": 13,
  "section": { "type": "superellipse", "n": 3.2 },
  "note": "measured with callipers at mid-fuse"
}
```

Section types are `superellipse` (with `n`), `rect` (with `fillet` in mm), and `naca` (with `flip`).
