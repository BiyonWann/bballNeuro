# Chat log #3

---

## User

Our data has LOC_X and LOC_Y in tenths of a foot, with the basket at (0, 0). Help me picture the court in these coordinates. Where are the corners, the top of the key and the baseline?

We split train and test by game instead of by row. Why does that matter? If I split by row, what would leak between train and test?

For the polar version we use distance and 23.75 times the angle. Why multiply the angle by 23.75 instead of using radians directly?

Shots right behind the rim wrapped to almost ±180 degrees and created a fake cell, so we clip the angle. Can you explain why that wrap happens, and why clipping a little past the baseline is enough?

---

## Claude

The diagram above shows the court. Here are your four questions in order.

1. Reading the court  
Units: 10 units = 1 ft. The court is 50 ft wide, so LOC_X runs from −250 to 250. The basket sits 5.25 ft in from the baseline, so the baseline is at LOC_Y = −52.5. Half court is 47 ft from the baseline, so it’s at LOC_Y = 417.5.  
Corners: The corner three is the straight part of the line at x = ±220 (22 ft). It meets the 23.75 ft arc at about y = 89.5, which is roughly 14 ft up from the baseline. The court corners are (±250, −52.5).  
Top of the key: People use this name for three different spots. The free-throw line is at y = 137.5. The top of the free-throw circle is at y = 197.5. The top of the three-point arc, which is what most people mean by a “top of the key three,” is at y = 237.5.  
Things to check in your data:  
The widely copied draw_court plotting code puts the baseline at −47.5 instead of −52.5. That’s half a foot off and looks fine on a plot, but use −52.5 if you need the real geometry.  
People often say LOC_X is mirrored compared with the broadcast view. Don’t assume either way: cross-check against SHOT_ZONE_AREA if you have it, for example by confirming that “Left Side(L)” shots really have the sign you expect.  
Any LOC_Y above 417.5 is a backcourt heave.  
These numbers are for the NBA court. College and WNBA three-point distances are different.  
2. Why split by game instead of by row  
Shots from the same game aren’t independent. If you split by row, shots from one game end up in both train and test, and the model gets graded partly on games it has already seen. What would leak:

Game context: the same two teams, lineups, opponent defense, pace, referees and score situation.  
That night’s luck: if a team shot unusually well, the model learns “this game means made shots,” and that pays off on the test rows from the same game.  
The arena’s scorekeeper: coordinates are recorded by each arena’s own crew, and their habits differ. Shared crews let the model match a test shot to its game.  
Game-level features: anything like a player’s FG% for the game, or game totals, is computed from the outcomes themselves, including the test shot’s own result. That’s direct label leakage.  
Near-duplicates: putbacks and repeated shots from the same spot by the same player.  
The result is a test score that’s too optimistic. In real use you’ll be predicting games the model has never seen, and splitting by game tests exactly that. If you care about generalizing across time or to new players, split by date or by player instead.

3. Why multiply the angle by 23.75  
The scaling puts both polar features in feet. An angle θ in radians times a radius r is an arc length, so 23.75·θ is the distance in feet along the three-point arc.

Raw radians only span about ±1.8 for in-bounds shots, while distance runs from 0 to about 47 ft. Anything that treats the two axes as comparable would then treat one radian the same as one foot, even though one radian at the arc covers 23.75 ft of court. That includes grid cells, k-nearest-neighbor or kernel distances, spatial smoothing and penalties. With the scaling, a cell 2 units wide in both directions is about 2 ft × 2 ft at the three-point line.

The trade-off is that the scaling is only exact at 23.75 ft. Close to the rim it exaggerates sideways separation: at 5 ft, 0.1 rad is 0.5 ft of real court but counts as 2.4 ft. Choosing 23.75 makes the scale right where the important shot decisions happen.

4. The wrap behind the rim, and why clipping fixes it  
This assumes your angle is atan2(LOC_X, LOC_Y), so 0 points straight out from the basket. atan2 returns values from −180° to +180°, and the jump between them falls on the negative y-axis, directly behind the rim.

Take two shots: (+10, −50) and (−10, −50). They are 2 ft apart on the court, but their angles are about +169° and −169°. With the 23.75 scaling that’s about 140 ft apart in feature space. They also land at the far ends of the angle range where nothing else is, so they form their own sparse cell. That’s the fake cell you saw.

A small clip past the baseline is enough because the baseline is only 5.25 ft behind the rim. Any shot with a meaningful distance from the basket is in front of the baseline, so its angle stays inside a narrow band:

The deepest baseline corner, (±250, −52.5), is at about ±102°.  
The corner-three spot on the baseline, (±220, −52.5), is at about ±103°.  
Every in-bounds shot outside a few feet has an angle within about ±104°.  
Angles beyond that only come from shots in the small area right around and behind the rim: reverse layups, tip-ins and coordinate noise. At that range the angle carries almost no information, since direction becomes poorly defined as distance approaches 0, and the distance feature already says “at the rim.”

So clipping at something like ±105–110° leaves every real baseline shot alone and folds the rare behind-the-rim shots onto the edge of the valid range. They join real baseline cells instead of creating an isolated one. Left and right behind-the-rim shots still go to opposite edges, but those edges are real, populated cells, so the artifact is gone.
