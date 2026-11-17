# Chat log #11

---

## User

The 3-point line is 22 ft in the corners and 23.75 ft at the top. Our shape test checks whether corner cells settle closer to the basket than top-of-the-arc cells. In 2024-25, in every run where both groups exist, they do: about 21.8–23.1 ft for corners against 25.3–26.4 ft for the top, so corner minus top is about −3.3 ft. Why is that gap bigger than the line’s own 1.75 ft difference?

The proposal said averaging shots along a curved arc pulls a prototype inside it, and polar inputs (distance and angle) should fix that. In 2024-25 with 6 cells, cartesian arc cells sit 0.66 ft inside the average distance of the shots they win, while polar cells sit just 0.04 ft off. Walk me through why averaging x and y coordinates along a curve lands inside the curve.

---

## Claude

Your two questions turn out to be connected, so I’ll take the newer one first.

Why averaging x and y along a curve lands inside it

Start with two shots on the same arc, at radius r, on either side of the top. Their average is the midpoint of the straight chord between them, and every chord of a circle lies inside the circle. Each extra shot you add is just more chord-averaging, so the mean can’t reach the arc.

You can also think in vectors from the basket. Each shot is an arrow of length r. The arrows point in different directions, so their sideways parts cancel when you add them. Only the parts along the cell’s central direction survive. So the length of the average arrow is smaller than the average arrow length.

For shots spread evenly over an angle of ±θ around the cell’s center, with points r(cos φ, sin φ):

The y-components cancel, so the mean is (r · sin θ / θ, 0).  
sin θ / θ < 1 for any θ > 0, so the prototype lands at r · sin θ / θ, which is inside the arc.  
The shortfall is about r · θ² / 6. It grows with the square of the cell’s angular width.

You can check this against your numbers. At r = 23.75 ft, a 0.66 ft shortfall needs θ ≈ 0.41 rad, which is about ±23°. So each arc cell spans roughly 47° of the arc. That’s believable with only 6 cells.

The general version doesn’t need equal radii. Distance is the length (norm) of the shot vector, and the norm is convex. Jensen’s inequality then says: length of the mean ≤ mean of the lengths. The two are equal only when every shot points in exactly the same direction from the basket.

Polar inputs avoid this because the prototype averages distance directly. Its distance coordinate is just the mean distance, with no cancellation. The 0.04 ft that’s left is not geometric. It’s probably from online learning: a finite learning rate means the prototype tracks recent wins rather than sitting at the exact mean. (Input scaling could also add a little.)

Why the corner-vs-top gap is about 3.3 ft, not 1.75 ft

A prototype sits where the shots are, not where the line is. Its distance is roughly the line distance plus how far past the line players actually shoot. That extra distance is very different in the two places:

Corners have a wall. The sideline is 25 ft from the basket sideways and the corner line is at 22 ft, so there’s only about 3 ft of floor. Corner threes are mostly catch-and-shoot spot-ups with toes on the line. There’s almost no room to drift deeper.  
The top has no wall. Step-backs, transition pull-ups, deep shooters and end-of-clock heaves all add a long tail out to 26–30+ ft. That pushes the mean about 2 ft past 23.75.

So the gap is about 1.75 + (extra depth at the top − extra depth in the corner). With roughly 2 ft at the top and roughly 0.5 ft in the corner, that comes to about 3.3 ft.

The chord effect actually works against this gap. In cartesian runs it pulls the curved top cells inward, which shrinks the gap. From the basket, the corner segment is a straight line at 22 ft, so it isn’t pulled in the same way. The gap shows up anyway.

Two things worth checking:

Confirm it without the model. Shotdetail has SHOT_ZONE_BASIC and SHOT_DISTANCE. Compare the mean distance of corner 3 attempts with above-the-break 3 attempts for 2024-25. If the gap is about 3 ft there too, the prototypes are just tracking the data.  
The 21.8 ft corner value is below the 22 ft line. That likely means a corner cell is also winning some long 2s along the baseline, because cell boundaries don’t follow the 3-point line. Counting the 2s versus 3s each corner cell wins in that run would tell you.

---

## User

Our zone maps show straight borders between cells with x/y inputs and curved borders with polar inputs. Why does each input type produce that border shape?

---

## Claude

Each map is a Voronoi diagram, and its borders are straight lines in input space. Encoding changes where the borders sit on the court, not the rule. Cartesian inputs are the court itself, so the lines stay straight. Polar inputs bend those lines when you draw them back on the court.

The border is always straight in input space

With winner-take-all, each shot goes to the nearest prototype. The border between cells 1 and 2 is the set of inputs u that are equally close to both:

‖u − w₁‖² = ‖u − w₂‖²

Expand both sides and the ‖u‖² terms cancel:

2u · (w₂ − w₁) = ‖w₂‖² − ‖w₁‖²

That equation is linear in u. So the border is a straight line in input space: the perpendicular bisector of the segment between the two prototypes. Every border in your maps comes from this. What differs is what “a straight line in input space” looks like on the court.

Cartesian: the input space is the court

With (x, y) inputs, a straight line in input space is a straight line on the court. The borders are perpendicular bisectors between prototypes, drawn right on the floor.

Polar: straight in (r, θ), curved on the court

With (r, θ) inputs, each border is a line a·r + b·θ = c. Mapped back to the court, there are three cases:

b = 0 → r = constant. The two prototypes have the same angle and different distances. The border is a circular arc around the basket, the same shape as the 3-point line.  
a = 0 → θ = constant. The two prototypes have the same distance and different angles. The border is a straight ray out from the basket.  
Both nonzero → r grows linearly with θ. The border is a piece of an Archimedean spiral. Most of your curved borders are probably this case, since prototypes rarely line up exactly in distance or angle.

So polar maps aren’t all curves. A border between two cells at about the same depth should look like a straight spoke through the basket. Input scaling (say, θ in radians vs degrees, or z-scored inputs) changes the weights a and b. That tilts how much each border is an arc versus a spoke, but never makes a border straight on the court unless it’s a spoke.

Why this matters for your project

This also explains the 21.8 ft corner cell from before. A cartesian border is a straight line, so it can’t follow the curved 3-point line. A cell on one side of the arc will always win some shots from the other side, like long 2s near the baseline. A polar r = constant border can sit right on the arc, so polar cells can split 2s from 3s cleanly.

The corners are the exception. There the 3-point line is straight and 22 ft out to the side of the basket, not a circle around it. So even polar borders only approximate it in the corners.
