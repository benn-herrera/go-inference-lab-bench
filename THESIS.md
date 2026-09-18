# THESIS – Inference Lab Bench

*Why this project has the shape it has*

---

## The thesis

**Many projects implement inference. None are built to make it comprehensible.**

Production engines are shaped by the demands of serving: throughput, memory
ceilings, batch efficiency. Those demands are legitimate, but the trade-offs all
lead toward opacity. The result runs a model well and explains nothing about how
it ran.

What this leaves missing is not documentation. It is that there is nowhere to
stand between the paper and the production kernel — no artifact in which the
mechanism of GPU-accelerated inference is both visible and real.

**The breadboard, not the sealed unit.**

A breadboard build has every connection exposed and reachable: any node can be
probed, any trace cut, any component swapped — and it still performs its
function at usable speed. That combination is the whole point. A schematic you
can read but cannot run teaches less than it appears to. A sealed unit that runs
fast teaches nothing at all.

This project is the breadboard version of an inference engine — built for
inspection, for visualization, and above all for modification, while remaining a
real engine with usable performance characteristics.

## What follows

**An architecture should be something you see, not something you find.** Model
architecture is therefore not only data, but visible data, and the engine executing
it is generic.

**Inspection is a feature, not a debugging affordance.** Visualization and the
capture of intermediate state are part of what the project is for. They are
designed, not bolted on where someone needed them once.

**Performance is a constraint, not a goal.** It has to stay good enough that
experiments run at conversational speed on real models, because a bench too slow
to iterate on stops being used. It is never a reason to make the mechanism
harder to see.

**A bench you cannot trust teaches you wrong things.** Numerical behaviour is
verified against an independent reference implementation. An inspectable engine
that quietly computes something subtly wrong is worse than no bench at all — it
produces confident understanding of a process that isn't happening.
