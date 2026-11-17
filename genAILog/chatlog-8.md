# Chat log #8

---

## User

Our shot-distance histogram has a big spike in the 0–0.5 ft bin in 2000-01 and 2008-09, and about 21.6% of 2000-01 shots are logged at exactly (0, 0). What’s the most likely reason, and how would it affect a network that starts its cells at random shots?

---

## Claude

The most likely cause is a placeholder location in older play-by-play logging, not a real shot pattern.

Why it’s probably an artifact

In the NBA stats shot data, LOC_X/LOC_Y are measured from the center of the hoop, so (0, 0) is the rim. When a scorer logged a shot without a precise spot, the location seems to have defaulted to the basket. This happened most often with layups, dunks and tip-ins, and probably with some missing coordinates too.  
Real shots at the rim don’t stack on one point. In seasons with good tracking, close shots spread across roughly 0–4 ft. One-fifth of a season landing at exactly (0, 0) points to a recording convention.  
That would explain why only some seasons spike. 2000-01 and 2008-09 probably had different logging practices or gaps in the source data. I haven’t confirmed the cause for each season, so treat that as a guess to check.

How to confirm it in a few minutes

Cross-tab the (0, 0) rows by ACTION_TYPE and SHOT_ZONE_BASIC. If they’re nearly all layups and dunks in the Restricted Area, the shots were real but their positions were collapsed to the rim. If jump shots or Mid-Range zones show up at (0, 0), those rows are missing locations.  
Compare the SHOT_DISTANCE column with the distance computed from the coordinates. A row with distance 12 at (0, 0) means the location is missing.  
Plot the share of shots at (0, 0) for every season. That shows which seasons are affected and whether the change is sudden.

Effect on your competitive learning network

Duplicate starting cells. If you start cells at random shots, about 21.6% of them will begin at exactly (0, 0) in 2000-01. Duplicate cells are always tied, and argmin breaks ties by index, so only the lowest-indexed one can ever win. The rest never move and become dead units. With 20 cells, about 4 are wasted from the start.  
Too much weight on the rim. One cell gets pinned to the rim and takes about a fifth of all updates. The other cells have to cover the rest of the court, so the perimeter and mid-range get coarser coverage.  
Distorted comparisons between seasons. If you compare learned layouts across seasons, 2000-01 and 2008-09 will look much more rim-heavy than they really were. That’s a data artifact, not a change in strategy.  
Breakage in the normalized version. If you use the classic dot-product version with normalized inputs and weights, (0, 0) is a zero vector. Normalizing it divides by zero and gives NaN, and its dot product with every cell is 0, so every cell ties. Euclidean distance avoids this.

Fixes

Start cells from unique locations, or from shots that aren’t at (0, 0). A k-means++-style spread also works, and so does adding a little jitter so no two cells start at the same point.  
Decide how to handle the (0, 0) rows based on the cross-tab:  
Mostly real rim shots: keep them, but count them as a separate “rim” category instead of letting them shape the map.  
Missing locations: drop them.  
Whatever you choose, apply the same filter to every season so the comparisons stay fair.

---

## User

Our shape test only counts cells beyond 20 ft. It calls a cell a corner cell if it’s at least 1.2 radians (about 69°) from straight-on, and a top cell if it’s within 0.4 radians (about 23°). Help me picture those cutoffs on the court. What part of the arc falls in neither group?

With 6 cells, 2024-25 gives corner cells at about 21.8 ft but no top cells, because the wing cells sit 25–31° off straight-on, just past the cutoff. Is the test failing, or is it showing something real about where the network puts its cells

---

## Claude

Here’s the court with those cutoffs shaded, plus your 2024-25 cells. The angle is measured at the hoop, from the line pointing straight toward half court, and only the area past 20 ft is shaded.

---

## User

With polar inputs (distance and angle), the corner cell sits at 23.1 ft against 24.8–25.5 ft for the other arc cells. Why would polar inputs move the corner cell outward compared with x/y inputs?

---

## Claude

What falls in neither group

The gap is the two wings above the break, from 23° to 69° off straight-on on each side, past 20 ft.

Corner cutoff: the corner three line meets the arc about 67.9° (1.185 rad) off straight-on, so your 1.2 rad cutoff sits about 1° past that point. The corner group is basically the straight corner strip, plus long twos along the baseline beyond 20 ft.  
Top cutoff: the edge of the top band crosses the arc about 9.3 ft to each side of center, a little outside the elbows.  
Size of the gap: each wing gap spans 0.8 rad, the same angle as the whole top band. On the 23.75 ft arc that’s about 19 ft of line per wing, against 19 ft for the top. So about two-thirds of the arc above the break counts as neither.

The missing top cell in 2024-25

The test isn’t broken, but it isn’t telling you top shots are rare either. It’s mostly reflecting how few cells cover the arc.

Why the wing cells land where they do: each cell settles at the average of the shots it wins. With 6 cells and a left–right symmetric shot chart, the arc above the break probably gets one cell per side. The straight-on shots get split down the middle, so neither cell can sit at 0°.  
What a flat spread would give: if shots were spread evenly from 0° to about 68° on one side, that side’s average would point at half the range, about 34°. Your cells are at 25–31°, so they’re already pulled toward the middle. That suggests the top is well used, not empty.  
The underlying issue: the test classifies cells by where their centers sit, and the average of a wide wedge always lands inside it. You’d see “no top cell” even if straight-on threes were common.

Ways to check:

Rerun with an odd number of cells (5 or 7) and with more cells (8–12), and see whether a cell appears near 0°.  
Plot a histogram of shot angles for 2024-25 threes above the break. That’s the direct answer to whether the top is popular.  
Classify by shots instead of cell centers: for each cell, count how many of the shots it wins fall in the top band and how many in each wing.  
Repeat with several random seeds, since one run can land differently.

Why polar inputs move the corner cell outward

In x/y, a cell’s distance from the hoop is the length of the average of its shot positions. The length of an average can’t exceed the average length, so the x/y center always sits at or inside the average shot distance. The more spread out in angle a cell’s shots are, the more it gets pulled in. In polar, the distance coordinate is just the average shot distance, with no pull inward.

The two input types also split up the court differently:

In x/y, the corner cell competes with nearby cells by straight-line distance. It can win baseline long twos from 19–21 ft, which lowers its average. I’m not sure how much of the 21.8 ft comes from this.  
In polar, the borders are drawn in distance–angle space, so which shots the corner cell wins depends on how you scale feet against radians. If you didn’t standardize the two inputs, distance dominates. Then the corner cell mostly collects shots at similar distances, wherever they are on the arc.

23.1 ft is close to where corner threes are actually taken: a step behind the 22 ft line, and up to about 23.75 ft near the break. The other arc cells sit at 24.8–25.5 ft because the line there is 23.75 ft and deep threes push the average out. So the polar number is probably the more faithful one here, and the x/y number is pulled in.

To check, take the shots the corner cell wins in each run and compare their average distance with the cell’s own distance. In x/y, that difference is the pull inward. Comparing which shots each version assigns to the corner cell shows how much comes from the different splits. Before trusting the polar layout, make sure distance and angle are standardized.
