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
3. The resulting section drawn to scale, against one baseline of your choosing, with water
   actually flowing over it.
4. Pressure distribution and boundary layer for that section.
5. An optimiser, described below.
6. A pump-cycle model that works out what angle the fuse actually sees.
7. Fuse drag against pump angle for every shape at once, with the pump cycle's range shaded.
8. All six shapes cut at the same angle on a common chord axis, with a table.
9. A line-up stage: a protractor-style angle dragger, CAD fashion, and every chosen fuse drawn
   side by side on one scale. Each tile pairs the fuse's own cross-section with the slice the
   water crosses at that angle, plus chord, t/c and drag.
10. The cross-sections in the line-up are editable, surfboard-template style: bezier anchor
    points with tangent handles. Drag a point, drag the handles to set the angle through it,
    click the outline to add a point, double-click to remove one. Sculpting the working fuse
    reshapes it across the whole playground; sculpting a preset makes a private copy with a
    reset.

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
| 7.8 mph (3.5 m/s) | 8.9° |
| 10.1 mph (4.5 m/s) | 5.4° |
| 15.7 mph (7.0 m/s) | 2.3° |

Those come from `foil-rl-pump/python-rl/foil_env/foil_physics.py` in this repo, whose
`compute_trim_angle` solves the full wing, stabiliser and mast balance. The playground reimplements
just the wing term and lands within about 2.5% of it.

On top of trim, the pump swings the flight path by `atan(A·ω/V)`, which is large: 80 mm of heave at
2 Hz and 10 mph is ±12.6°. The rider feathers most of that out by pitching with the stroke, and
what is left is the swing the fuse actually works through.

The headline is not that pumping adds angle. It is that **trim itself is 5 to 9° at pumping speeds**,
which is far more than the one or two degrees most people would picture. Slow, high-lift pumping is
exactly where the fuse is most badly aligned.

## Drag estimates

**Corrected after an independent review.** An earlier version cut the body into streamwise strips
and charged each one a full-dynamic-pressure base drag. That was wrong twice over. The strips are
not high-aspect two-dimensional sections: at 600 mm long, 38 mm tall and 10°, the run across the
cuts is 142 mm against a 219 mm chord. And the flat faces were billed at `q` when the crossflow only
reaches them at `q sin²α`, with the resulting force projecting through another `sin α`. That is a
factor of about 33 at 10°, and it fell hardest on exactly the blunt shapes the tool was judging.

The model now resolves the fuse as a slender body at incidence, which is the standard treatment:

- **Axial**: skin friction over the true wetted surface at the fuse's own Reynolds number, with a
  strut form factor on the plan thickness ratio. Nearly independent of angle.
- **Crossflow**: `V sin α` meets the cross-section as a 2D shape of chord `h` and thickness `w`,
  producing a normal force whose streamwise component is drag. Scales as `sin³α`, so it fades fast
  with speed. The coefficient comes from fineness ratio `h/w`, fitted to elliptic cylinders, times a
  penalty for how much width the outline retains at top and bottom.

**Section solve.** The selected streamwise cut still goes through a constant-strength source panel
method, then Thwaites, Michel transition, Head's method and Squire–Young. That is a genuine 2D solve
of that shape and is what shows separation.

**The animation.** A D2Q9 lattice Boltzmann solver with pulsed dye. Notes on how it works, and the
three bugs found building it, are below.

**Still approximate.** The finite-length crossflow factor is 0.75, inside the usual 0.7–0.8 band but
not derived. The corner penalty is a fitted curve, not a measurement. Nose, tail and mast junctions
are not modelled. Read differences between outlines; treat absolute newtons as indicative.

## What changed, and what it cost

The headline number moved a long way under review:

| Claim | Before | After |
|---|---|---|
| Aerofoil vs optimised rounded rect | 64% less drag | **21% less drag** |
| Typical fuse at 17.4°, 8.9 mph | ~26 N | **~9 N** |

Two outright bugs were also found and fixed. `shapeIntegrals` evaluated every outline at a hardcoded
nominal size, so a rounded rectangle's area and stiffness described a different shape than the one
being drawn (correct only at 38 × 13). And the second moment was taken about mid-height rather than
the centroid, which overstated the stiffness of any outline whose centroid sits off centre — an
aerofoil standing on end being exactly that case, by 11%.

The optimiser also now constrains lateral stiffness. Vertical stiffness alone goes as `w·h³` and
gets cheaper the taller the fuse, so it drove the search to a 6.5 mm wide blade. Holding area, `Iyy`
and `Izz` all at once leaves only the design you started with, so how much lateral stiffness you
will give up is exposed as a control. The answer moves a lot with it.

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

Section types are `superellipse` (with `n`), `rect` (with `fillet` in mm), `naca` (with `flip`),
and `custom` (with `pts`: bezier anchors as `{z, w, iz, iw, oz, ow}` in normalised height and
half-width, the format the sculpting editor writes).
