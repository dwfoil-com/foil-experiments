# Hydrofoil Forces and Structure

This project should include a structural-load section, because many rider intuitions about "stiff", "direct", "dead", or "breaking boxes" are really about leverage and load paths, not just hydrodynamics.

## Core Idea

The mast is a long lever arm. Rider weight and pump inputs create large moments at the mast base, which resolve into:

- Compression near the front of the mast tracks / plate interface
- Tension near the rear of the mast tracks / bolts
- Shear through the mast base and hardware
- Local deck and bottom skin stresses in the board sandwich

Repeated cyclic loading matters even when nothing fails dramatically.

## Example Forum-Derived Load Case

One useful illustrative case from builder discussions is a front-foot-loaded pump from a roughly 100 kg rider. The reported free-body estimate was approximately:

- 493 kg equivalent compression at the front bolt / front attachment region
- 393 kg equivalent tension at the rear bolt / rear attachment region
- Front reaction occurring only about 65 mm forward of mast center

Whether those exact numbers hold for every setup is less important than the teaching point: the moment arm is short at the plate, long through the mast, and load multiplication at the board interface can be surprisingly high.

## Educational Claims To Cover

### Front-foot-loaded pump spikes board loads

One useful rider-facing explanation is that a hard front-foot pump does not just "load the foil". It also drives a large moment into the mast base, like loading a diving board. That can create very high compressive force at the front attachment region and strong rear tension at the back.

### Failure is often gradual

The educational article should distinguish:

- Catastrophic box failure
- Bolt loosening
- Compressive crushing
- Rear pull-through / tension damage
- Hidden flex and loss of responsiveness before visible failure

### Flex matters for feel

Especially for prone and dock-start pumping, a board can feel dead or vague before it obviously breaks. Loss of stiffness can reduce pumping response even if the box is still technically intact.

### Bigger plate area can matter

The article should also mention that mast plate area changes how loads spread into the board. A larger plate does not remove the global moment, but it can reduce local stress concentration and change how "direct" or "supported" the setup feels.

## Simplified Model To Add

Useful first-pass structural view:

- Rider load applied at stance location
- Mast as lever to the tracks
- Plate / bolts represented as front and rear reactions
- Output approximate front compression and rear tension forces
- Optional plate-area parameter to show load spreading

This does not need finite element detail to be educational. A clean statics model is enough for the article.

## Questions Worth Answering

- How do stance and pump style change reaction loads at the mast base?
- How much does mast length increase moments into the board?
- Why do front compression and rear tension show up so often in failures?
- Why can a board "lose pump" before it visibly breaks?
- How do dense inserts, PVC / Divinycell reinforcement, and sandwich thickness change the picture?
- How much does mast plate area change local stress and perceived stiffness?

## Article Treatment

The best format is probably:

1. Free-body diagram of rider, mast, tracks, and reactions
2. Slider for rider mass, stance, mast length, and pump impulse
3. Visual stress zones in the board around the boxes
4. Short builder notes on stiff inserts, skins, and load spreading

This section should stay scoped to explanation, not manufacturing advice or safety claims.
