# The general interface operator as vecdiff's engine

### How the method we built extends a Cartesian-oval solver into a general vectorial-diffraction package

The thing we actually built this session is not another focusing code. It is a
*method*: a way to compute the vector field scattered by an **arbitrary smooth
dielectric interface**, written so that the interface is an operator on the
angular spectrum rather than a formula for a focus. That distinction is the whole
argument for unification. vecdiff, today, is a superb instrument for one surface
— the stigmatic Cartesian oval — solved rigorously by a Hankel-spectral
propagator. The general operator is the engine that lets the same package answer
the question for *any* surface, and, because the operator composes, for *systems*
of surfaces. This essay is about making that operator the heart of vecdiff, with
everything vecdiff already does well arranged around it.

## What the method is, in one breath

A field in a homogeneous medium is its angular spectrum. A plane interface acts
diagonally on that spectrum — one Fresnel dyadic per direction. A curved
interface does not, and the method computes the non-diagonal operator that
replaces it, in three moves. Locally, at each point of the surface, the field is
refracted as if by the tangent plane: read the incident ray direction from the
field's own local wavevector, split into s and p against the local plane of
incidence, apply the vector Fresnel amplitudes. Globally, the refracted surface
field is radiated back into an outgoing spectrum by a Franz–Stratton–Chu surface
transform — realised two ways, an azimuthal Bessel kernel when the surface is a
body of revolution and a general NUFFT of the surface currents when it is not.
And the state is carried on the **sphere of directions**, where the projection
Jacobian is unity, the two artificial horizons of a Cartesian chart disappear,
and — the payoff — every interface maps the same space of states to itself, so
interfaces *compose*. A multi-surface system is their ordered product with free
propagation between; a closed body is the same product with one surface repeated.

That is the engine: local Fresnel, a surface transform, and composition on the
sphere. Its cost is leading order in one over the size parameter, which is
exactly the regime — interfaces hundreds of wavelengths across — where rigorous
volume solvers cannot go and where the approximation becomes accurate.

## Why it is the right engine for vecdiff specifically

The reason this is a *natural* extension and not a bolt-on is that vecdiff's
whole domain is the special case where the operator is exactly solvable. Feed the
general operator the stigmatic Cartesian oval and it collapses to the aplanatic
transfer — the √cosθ apodization the intrinsic Jacobian supplies is the Abbe
sine condition, and the polarization eigenvalues reduce to the familiar
(cosθ, 1, sinθ). vecdiff's focus is the value the general operator takes on
vecdiff's surface. So adopting the operator as the engine costs vecdiff nothing
it already has, and buys it every surface it currently cannot touch: the sphere,
the paraboloid, the over- and under-corrected conics, polynomial aspheres, and
genuine freeforms through the NUFFT path. More than new shapes, it buys
*systems*. vecdiff propagates through one diopter; the operator composes, so two
diopters, a thick lens, a folded objective are words in the same algebra rather
than new derivations. That is the step from "a solver for the Cartesian oval" to
"a package for vectorial diffraction."

The honest counterpart to that reach is a loss of rigor, and it should be stated
plainly because it dictates the architecture. vecdiff's Hankel-oval propagator
is, within its class, essentially exact. The general operator is leading order in
1/kR: it represents the partial waves that reach the surface geometrically and is
blind to the sub-wavelength, tunnelling band with transverse order above kR. On a
large interface that band shrinks with size and the approximation is excellent;
on a small or extreme one it is not. So the general operator does not *replace*
vecdiff's exact engine — it surrounds it. Where the surface is a body of
revolution, vecdiff's Hankel path is the faster and more rigorous backend and
should be the default; where it is not, the general operator is the only option
and carries the load. They are the same physics at two fidelities, and the
package should let the user pick, and — crucially — check one against the other
on the overlap.

## The pieces of vecdiff, rearranged around it

Almost everything vecdiff already has becomes more valuable once the general
operator is the core rather than a competitor.

The **Stratton–Chu solver** you keep as a convention-arbiter becomes the
package's built-in referee. The general operator is an approximation; the honest
way to ship an approximation is with the exact answer next to it on the one
geometry where the exact answer is tractable. Running SC against the operator on
the stigmatic case, in CI, on every commit, is what earns the operator the right
to be trusted on the cases where no exact answer exists. Mie does the same for
the sphere encounter-by-encounter; a full-wave reference does it for the
non-separable cases at modest size. The operator brings the generality; vecdiff's
referee keeps it honest.

The **Hankel propagator** is not superseded; it is promoted to the fast,
rigorous, axisymmetric backend behind the same operator interface, and it remains
the independent witness that validates the general path. Keeping two engines is a
deliberate cost paid for that independence: two engines agreeing is evidence in a
way one engine never is.

The **field containers, the polarization map, the diagnostics and the animation**
are engine-agnostic and become the shared substrate that renders whatever the
operator produces. The one design decision is that the unified field must carry
both notions the two engines need — the reference surface a sample lives on, and
the angular spectrum it can be reduced to — a superset of the two field models,
not an adoption of either.

## Reading real systems into the operator

The generality of the operator is what finally makes it worth wiring the ray
tracer to the front. Because the operator accepts *any* incident spectrum and
composes across surfaces, it can consume the exit-pupil state of a real optical
system rather than an idealised pupil. The tracer reads a prescription, solves
each surface exactly, recovers the chief ray and object plane, and delivers the
exit-pupil wavefront, apodization and ray-mapped polarization as one bundle —
precisely the incident state the operator needs. The division of labour is forced
by scale: geometry carries the field through the hundred-million-wavelength bulk,
and the operator does vectorial diffraction on the last microns where
polarization decides the focus. The bridge is a single `ExitPupil` object the
tracer fills and the operator consumes, governed by the convention contract that
fixes what "wavefront" and "apodization" mean as they cross — the same seam where
the reference-sphere and units bugs lived, so fixing those with tests *is*
building the bridge. Done once, the package ingests a patent objective and
returns the vector PSF, the aerial image and the polarization map end to end, and
the HDR viewer shows you the hardware the field came from.

## What the result would be

A package whose core is a general, composable, vectorial interface operator;
whose fast rigorous special case and built-in Maxwell referee are vecdiff's
existing engine and solver; whose front door is a real ray tracer; and whose
output is the vector field, the aerial image and the polarization of an actual
optical system. No open tool occupies that whole span. rayoptics stops at
geometry; PyFocus does the aplanatic focus of an ideal pupil; prysm and diffractio
do general scalar propagation; the full-wave solvers are exact but capped at tens
of wavelengths. The operator is what stitches "any surface, composed, at large
scale" to "vectorial and verified," and vecdiff already holds the rigorous anchor
and the diagnostics that make it trustworthy.

## The path, method-first

Lead with the engine. First, lift the general operator — `InterfaceOperator`,
`FreeSpace`, `System`, the Bessel and NUFFT return integrals — into vecdiff behind
an operator interface that vecdiff's Hankel propagator also implements, so both
are callable the same way and the oval case has two independent backends from day
one. Second, wire the Stratton–Chu solver as the automatic referee in CI, so the
general operator is validated against the exact field on the stigmatic surface on
every change. Third, unify the field and lift vecdiff's diagnostics and
polarization map onto it as the shared substrate. Fourth, fix the tracer's
exit-pupil seam and build the `ExitPupil` bridge, and prove the whole chain on a
patent system. The generality is the headline — vecdiff, past the Cartesian oval,
onto any surface and any system — and the rigor vecdiff already has is what makes
that headline safe to print.

The one line to keep in view: the method does not make vecdiff less itself. It
makes vecdiff's one exactly-solved surface the trusted anchor of a solver for all
the others.
