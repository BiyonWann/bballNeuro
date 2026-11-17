# Chat log #12

---

## User

My idea is to see if a competitive learning network can find the NBA 3-point line using only where shots were taken, without ever being told where the line is. Over the last 20 years teams stopped taking long twos, so shots pile up at the rim and just behind the arc. Before I commit to this, is that a reasonable thing for competitive learning to pick up on? What would the network actually be detecting?

---

## Claude

Yes, it’s reasonable, but the network won’t find the line directly. It will find where the shots are, and the line shows up as the inner edge of one of those clusters.

What competitive learning detects

Winner-take-all learning on (x, y) shot locations is online vector quantization. Each unit’s weight vector moves toward the shots it wins, so the units end up sitting in dense regions. With many units, they spread out roughly in proportion to density. The network is mapping where shots come from. It has no notion of an edge.

The 3-point line is an edge: shot density jumps from sparse (the long-two zone) to dense just behind the arc. So with enough units you should see:

A cluster of units at the rim and in the paint  
A ring of units along the arc, about 1–3 ft outside the line, because players set up with their toes just behind it  
Few or no units in the long-two zone in recent seasons

The line is the inner edge of that ring. What you’re really recovering is how the line changes player behavior, not the painted line itself. That’s still a fine framing for a comp neuro project: the network learns a structure in the world only through how that structure shapes the input statistics.

Where it could go wrong

Too few units. With about 5–10 units, the boundary between the rim units and the arc units lands halfway between them, around 12–15 ft, nowhere near the line. You need enough units to trace a ring.  
The rim takes over. Rim shots are a big share of attempts. Units can collapse there, and some units never win anything (“dead units”). A “conscience” mechanism (frequency-sensitive competitive learning) or a Kohonen self-organizing map fixes this. A 1D chain SOM is especially nice because it tends to lay itself along the arc.  
The “why not a histogram?” objection. A histogram of shot distance alone shows a spike around 24–26 ft. The 2D version earns its place through the corners: the line is a 23.75 ft arc that turns into straight 22 ft segments along the sidelines. If your arc units flatten into straight lines in the corners and move inward, the network found the geometry, not just a radius. That’s your strongest test.  
Inputs. Give the network raw Cartesian coordinates. Polar distance from the hoop hands it the answer and turns the problem 1D.  
Outliers. Filter out heaves and backcourt shots, or they’ll grab units.

Ways to make the result stronger

Train per season. The long-two zone was well populated in the early 2000s, so the edge should get sharper over the years. You could show the arc ring forming over time.  
A natural control, if your data goes back that far. The NBA used a shortened, uniform 22 ft line from 1994–95 through 1996–97. If your shotdetail data includes 1996–97, the network should find a smaller circular ring with no corner break that season. Check the dates against your dataset.  
A number to report. Measure each arc unit’s distance from the hoop and compare it to the real line. You should see a consistent outward offset in feet. Explaining that offset (players stand behind the line, not on it) is part of the result.

So it’s a sound project as long as you frame it as finding the shot-density band the line creates and inferring the line from its inner edge. The corner geometry is what turns it from a trivial result into a convincing one.

---

## User

ok so if each cell’s weights end up at the center of the shots it wins, I can plot every cell right on the court. In our course notes the winner is the cell with the biggest net input, the sum of each weight times its input. For court coordinates, why would that pick the wrong cell? Should I use Euclidean distance instead?

---

## Claude

Yes, use Euclidean distance. The dot-product rule only works when inputs and weights are normalized, and normalizing court coordinates would erase the information you care about.

Why the biggest net input picks the wrong cell

Net input is w · x = |w| |x| cos θ. It rewards two things: pointing in the same direction as the shot, and having long weights. It doesn’t reward being close to the shot.

Take the hoop as the origin. That’s how stats.nba’s LOC_X/LOC_Y data is set up, in tenths of a foot. Say one cell sits at the rim, w = (0, 3), another sits at the top of the arc, w = (0, 25), and a shot comes from 5 ft straight out, x = (0, 5):

Rim cell: 3 × 5 = 15  
Arc cell: 25 × 5 = 125

The arc cell wins even though the shot is 2 ft from the rim cell and 20 ft from the arc cell. In general, cells with long weight vectors win everything in their direction, and cells near the hoop almost never win. With the dot product, the cells divide the court into angular wedges, not into nearby neighborhoods of shots.

Why your course notes get away with it

The notes probably assume every input and weight vector has length 1. Expand the squared distance:

|x − w|² = |x|² + |w|² − 2 w · x

If |w| is the same for every cell, the cell with the biggest w · x is also the closest cell, so the two rules agree. Normalizing 2D court coordinates, though, projects every shot onto a unit circle and keeps only its angle. A layup and a 3 from the same direction become identical. That throws away the distance from the hoop, which is the whole 3-point signal.

What to do

Simplest: choose the winner by minimum Euclidean distance, and keep the update w += η(x − w). That update already assumes distance-based competition, and it’s what makes each cell’s weights land at the center of the shots it wins. That’s also what lets you plot cells on the court, because the weights live in court coordinates.  
To keep the “net input” story for the writeup: from the expansion above, picking the cell with the biggest w · x − |w|²/2 gives the same winner as minimum distance. That’s the net input plus a bias of −|w|²/2 per cell, so you can still describe it as a neuron with a bias.  
Another standard trick: add a third input component, z = √(R² − x² − y²), with R bigger than any shot distance. Every input vector then has the same length R, so normalized dot products work again and the distance information isn’t lost.

One practical note: LOC_Y goes slightly negative for shots from behind the backboard, and the hoop sits at (0, 0). Plot the cells in the same coordinates, then draw the court lines on top: a 23.75 ft arc and the 22 ft corner lines, which are 220 and 237.5 in tenths of a foot.

---

## User

How many cells should I start with? I was thinking 6, since the NBA has six official shot zones. But I’m worried the rim alone takes a big share of all shots and will eat up most of the cells. What would 6 cells probably look like, and is it worth also trying 12 or 20?

---

## Claude

Start with 6 so you can compare against the official zones, but plan on 12 and 20 (and probably 40) as well. With 6 you’ll mostly learn why the cells don’t match the zones. The arc shape only shows up at higher counts.

The rim probably won’t take most of the cells

With Euclidean competition, the cells end up minimizing squared distance from each shot to its winner. How many cells a region gets depends on how many shots it has and how spread out they are. Rim shots are numerous but packed within a few feet of the hoop, so one cell covers them cheaply. Three-pointers are spread along a band about 75 ft long, corner to corner, so that’s where extra cells pay off. Adding a second rim cell barely lowers the error, while adding another arc cell lowers it a lot.

So at equilibrium the arc gets most of the cells. The real rim risk is at the start:

Initialization. If you seed cells from random shots, roughly a third of seeds land at the rim. Several cells can crowd there, split it into tiny pieces, and stay stuck because nothing pulls them out. That’s a local minimum, not where the method naturally settles.  
Fixes: k-means++-style seeding (pick each new seed far from the existing ones), seeding uniformly over the half-court, or the conscience mechanism I mentioned earlier. Run several random seeds anyway and check that the layouts agree.

What 6 cells probably look like (recent season)

1 cell at the rim. The rest of the paint likely merges into it rather than getting its own cell.  
Around 5 cells along the arc: maybe both corners, both wings and the top. The corners are dense and separated from the rest of the arc, so they may well get cells of their own.  
No mid-range cell.  
The boundary between the rim cell and the arc cells falls halfway between them, around 12–14 ft, nowhere near the line.

So the 6 cells won’t recover the 6 official zones (restricted area, non-RA paint, mid-range, two corner 3s, above-the-break 3). The official zones are drawn by rules, and the cells follow where shots are dense. That mismatch is worth a paragraph in your writeup: modern shooting has merged the paint into the rim and nearly emptied the mid-range. Run the same 6 cells on an early-2000s season and a mid-range cell should appear. That’s a nice contrast.

Why go higher

12: about 9–10 cells on the arc. You start to see a curve, but each cell still covers a wide slice.  
20–40: a clear ring of cells, enough points to check whether the corners flatten into straight segments at 22 ft, and enough arc cells to measure the outward offset from the line reliably.

One subtle bias to watch for: a cell’s center is the average of a curved slice of the band, and that average sits slightly inside the curve, the way a chord sits inside an arc. With few cells covering wide slices, this pulls the arc cells inward, which partly cancels the “players stand behind the line” offset. It shrinks as you add cells, which is one more reason to report distances from the 20+ runs.

A clean way to present all this: sweep k = 6, 12, 20, 40, plot the error against k, and show a court plot for each.

---

## User

I need a way to tell whether the network actually “found” the line. My idea is to label each cell as a 2 or a 3 by what most of its shots are, then check that against real shots. But in 2000-01 only about 17% of shots were threes, so always guessing “2” would already be right most of the time. What baselines do I need so a high accuracy actually means something?

---

## Claude

You’re right that raw accuracy won’t mean much here. There’s also a second problem: even balanced metrics can look great when the cell boundary is nowhere near the line. So you need baselines that rule out both “the classes are imbalanced” and “any partition would do this.”

Why accuracy can mislead even beyond the 83%

Most shots are far from the line, either at the rim or well behind the arc, so almost any rim-vs-arc split labels them correctly. In a recent season, a boundary at 13 ft could still score very well, because almost nobody shoots from 13–23 ft anymore. In that case the network separated 2s from 3s without locating the line. In 2000-01 the same 13 ft boundary would label every long two a 3, so accuracy drops. That contrast is informative in itself.

Metrics to report instead of raw accuracy

Balanced accuracy, or Cohen’s kappa. Always guessing “2” scores 0.5 balanced accuracy and 0 kappa, whatever the 3-point share. This also makes seasons with different 3-point rates comparable.  
3-point recall and precision separately, so you can see which way the errors go.  
Accuracy near the line. Re-score only shots within about 3 ft of the line (on either side), and compute the majority-class baseline within that band too. This is the number that tests whether the line was found, since that band is where all the hard cases are.

Baselines to compare against

Majority class. Always guess “2.” This is your 83% floor in 2000-01. Balanced metrics already account for it, but report it anyway.  
Untrained cells. Same k, same initialization, no learning, labeled by majority vote the same way. This isolates what learning added. If trained cells barely beat their own starting positions, learning isn’t doing the work.  
Grid cells. Same k, placed on a uniform grid over the half-court, majority-labeled. This tests whether placing cells by density beats just having k regions. With large k, almost any partition gets good, so this baseline matters more as k grows.  
A single circle. One radius threshold, fit to the labels, ignoring the corners. If your cells don’t beat this near the corners, they haven’t learned the corner geometry.  
Supervised ceiling. The best accuracy a k-cell partition could reach when it’s allowed to use the labels, for example k-means initialization followed by label-aware tuning, or just the true rule. The true rule scores 100%, so a more useful ceiling is “best k-region partition.” Report where your network lands between the untrained baseline and this ceiling.

Two methodological points

Keep the readout separate from the test. Labeling cells by majority vote uses the true 2/3 labels, so do it on one set of games and score on held-out games, or label on one season and test on the next. Then say plainly in the writeup that the network never sees labels; only the readout does.  
Add a geometric check that uses no labels. Accuracy only scores the cell boundaries. You can also measure where the arc cells sit: their distance from the hoop compared with 23.75 ft, and whether the corner cells sit near 22 ft. A network that “found the line” should pass both: high balanced accuracy near the line, and arc cells that trace the line’s shape with a consistent small outward offset.

So you’d show a table per season and per k: balanced accuracy overall, balanced accuracy near the line, and the same two numbers for the untrained and grid baselines. Then add the geometric offsets next to it. If the trained cells clearly beat both baselines near the line in a recent season but not in 2000-01, that’s strong evidence the network is picking up the line through the behavior it causes.

---

## User

The 3-point line is 22 ft in the corners and 23.75 ft at the top of the arc. If the network really learns the shape of the line and not just a distance, corner cells should settle closer to the basket than top cells. How could I test that? Why might averaging shots along a curved arc pull a cell slightly inside the line, and would distance and angle inputs fix that?

---

## Claude

Your test works, but you should compare each cell to the line at its own angle instead of just checking whether corner cells sit closer. There are also two confounds: the sideline, and the arc-averaging bias you asked about. Both can be handled.

Testing corner vs. top

Convert cell centers to polar coordinates after training. For each arc cell, compute its distance from the hoop r and its angle θ, measured from straight out. The corner straightaway starts where √(22² + y²) = 23.75, about 9 ft above the hoop, which is an angle of about 68°. Below that point the line is the straight segment x = ±22 ft. Above it, the line is the 23.75 ft arc.  
Compute each cell’s offset from the line at its angle. The line’s distance is 23.75 ft above the break and 22 / sin θ in the corners (22 ft at the baseline, rising to 23.75 ft at the break). If the network learned the shape, the offset should be roughly the same everywhere. If it only learned “about 25 ft out,” corner cells will have bigger offsets than top cells.  
Fit two models to the arc cells: a circle (r = R) and the line’s shape plus a constant offset. Compare the leftover error. That’s a direct test of shape versus distance.  
Get uncertainty. You’ll only have one or two cells per corner, so use k = 30–40, pool the left and right corners, and resample by game. Retrain on each resample, or at least rerun with several random seeds, to get a confidence interval on the corner-minus-top difference. Your prediction is about −1.75 ft. A distance-only network predicts 0.

The sideline confound

The sideline is 25 ft from the hoop, so corner shooters are boxed into a strip between 22 and 25 ft. Above the break, shooters can drift out to 27 ft or more. So corner cells could sit closer just because of the sideline. To separate the two, also measure the inner edge of each arc cell’s shots: a low percentile, like the 5th, of their distance from the hoop. The sideline caps the outer edge, not the inner edge. So if the inner edge tracks 22 ft in the corners and 23.75 ft at the top, that’s the line, not the sideline.

Why averaging along a curve pulls a cell inward

A cell’s center is the average of its shots. If those shots lie along a curved band, the average lies on the inner side of the curve, the way a chord’s midpoint sits inside a circle. For shots spread evenly over an arc of radius R spanning an angle φ, the center lands at R · sin(φ/2) / (φ/2) from the hoop:

Arc slice per cell	Inward shift at 25 ft  
60°	about 1.1 ft  
36°	about 0.4 ft  
18°	about 0.1 ft

The shift grows roughly with the square of the slice angle, so more cells shrink it quickly.

This matters for your test because the corner segments are straight, and points along a straight segment average to a point on that segment. So the bias pulls the top cells inward but not the corner cells, which shrinks the corner-vs-top difference you’re trying to detect. That makes the test conservative: a difference that shows up anyway is real. Cells that straddle the 68° break are pulled inside the corner, so treat them separately or leave them out of the comparison.

Would distance and angle inputs fix it?

Partly, but they cause new problems:

They fix the top. In (r, θ) space the arc becomes a flat band, so averaging along it no longer pulls inward.  
They bend the corners. The straight corner segment becomes the curve r = 22 / sin θ, which bows outward. Averaging along it now pushes corner cells outward, so you swap one bias for another, though it’s small because the corners cover a narrow range of angles.  
They distort distance. One radian of angle is about 24 ft of court at the arc but only 3 ft at the rim. Euclidean distance in (r, θ) isn’t real distance unless you rescale.  
They bring back the rim problem. Near the hoop, a tiny move changes θ a lot, so the compact rim cluster stretches into a long strip across all angles. Then it really could take up many cells.  
They give the network the answer. Handing it the distance from the hoop weakens your “never told where the line is” claim.

What I’d do instead

Train on raw Cartesian coordinates, convert to polar only when analyzing the results, use enough cells that each arc slice is small (about 20° or less), and calibrate the method with a known-answer simulation. Generate synthetic shots lying exactly on the true line plus realistic noise, run the same pipeline, and measure the offsets it produces. Whatever offset shows up there is the method’s own bias. Subtract it from your real-data offsets to get the actual “players stand behind the line” effect.

---

## User

If I use distance and angle as the two inputs, distance is in feet but angle is in radians, so the angle would barely matter when the network measures closeness. How should I put them on the same scale? Is there anything odd that could happen with shots right behind the rim?

For the data, I’m using shot charts from about 200,000 shots a season. I want to split train and test fairly. Why would splitting by game be better than splitting by individual shot? And if I sample 20,000 training shots per season, does that make the eras fair to compare?

Results depend on where the cells start, and starting them at random shots will probably put a lot of them near the rim. How many times should I repeat each run, and what should I do if some cells never win any shots?

---

## Claude

Short answers: perfect scaling of distance and angle just gives you back Cartesian coordinates, so any polar scaling is a compromise. Splitting by game prevents near-duplicate shots from leaking between train and test, and a fixed sample size equalizes sampling noise but not shot mix or data quality. For seeds, plan on about 20–50 runs per setting. Dead cells are best handled by moving them to where the error is highest. Details below.

1. Scaling distance and angle

Multiply the angle by a reference radius. Use (r, R₀·θ) with R₀ = 23.75 ft. R₀·θ is arc length at the line, measured in feet. Near the arc, distance in this space is then close to real court distance, which is where you care about it.

You can’t get the scale right everywhere, because true distance in polar coordinates is ds² = dr² + r²dθ². The angle’s weight depends on r, and the only way to weight it correctly at every r is r·θ locally, which is just Cartesian coordinates again. So fixing R₀ means the angle is weighted correctly at about 24 ft, too heavily near the rim, and too lightly deep out. Treat R₀ as a free choice: try something like 12, 23.75 and 40, and report whether the conclusions change. Z-scoring each input is the other common option, but the scale it picks comes from the data’s spread, not from anything physical.

What goes wrong right behind the rim:

Angle wraps around. With θ measured from straight out, a shot just behind-left of the hoop has θ near +180° and one just behind-right has θ near −180°. They’re a foot apart on the court but about 2π·R₀ ≈ 150 ft apart in the inputs, so the same cluster gets split.  
Angle is meaningless near r = 0. A few inches of charting noise swings θ wildly, so the compact rim cluster smears into a long strip across every angle. That’s the “rim eats cells” scenario, now real.  
Pile-ups at exactly (0, 0). atan2(0, 0) returns 0 by convention, so any shots recorded exactly at the hoop all get θ = 0, an artificial spike. Count how many shots sit exactly at the origin.

You can patch these: encode the angle as R₀·(cos θ, sin θ) to remove the wraparound, and handle r below about 4 ft separately. Each patch moves you back toward Cartesian, though. So I’d still train on Cartesian coordinates and use polar inputs, if at all, as a comparison condition.

2. Train/test split and sample size

Why split by game. Shots within a game aren’t independent: same players, same defenses, the same arena’s scorekeeper charting locations (charting habits differ by arena), plus end-of-quarter heaves. If you split by individual shot, near-twins of your test shots sit in the training data, and the test mostly measures memorization of those games. Splitting by game asks the real question: does this generalize to games the network hasn’t seen?

Expect the difference to be small for the cells themselves, since 40 cells is only 80 numbers fit to 20,000 shots. It matters more for the majority-vote readout and the near-line band, where shot counts are small. If you want to go stricter, run two robustness checks: hold out whole arenas (removes scorekeeper effects) or train on the first half of the season and test on the second.

Is 20,000 shots per season fair? It equalizes sampling noise and the total amount of training, which is necessary. It isn’t sufficient:

Shot mix still differs. 20,000 shots from 2000-01 hold roughly 3,400 threes, versus several times that in a recent season. Early-era arc cells get fewer shots, so their positions are noisier. That’s the phenomenon itself, not unfairness, but it means early-era confidence intervals will be wider. Report them.  
Data quality may differ. Shot charting methods have changed over the years, and early coordinates may be noisier, which would blur the edge and could look like a behavior difference. Check it per season: what fraction of shots labeled 3PT sit inside 22 ft, or labeled 2PT sit beyond 24 ft? That gives you a noise floor for each era.  
Sample by game, not by shot. Draw whole games until you reach about 20,000 shots, so the sample keeps the same structure as your split.  
Hold everything else fixed: k, initialization method, number of updates, learning-rate schedule, and test-set size.  
Draw several samples. Different 20,000-shot draws tell you how much of a season difference is sampling luck.  
3. Repeats and dead cells

How many runs. Each run on 20,000 2D points takes seconds, so runs are cheap. Start with 20 per (season, k) setting. Plot your key statistic, such as the corner-minus-top offset, against the number of runs, and stop once its confidence interval stops shrinking meaningfully. Usually that’s 20–50.

Make each repeat use a new seed and a new game subsample, so the spread covers both sources of variation.

Comparing across runs. Cell numbering is arbitrary from run to run, so don’t compare cell 7 to cell 7. Compare run-level summaries: offsets, the corner-minus-top difference, balanced accuracy. If you need cell-by-cell comparison, match cells to a reference run by nearest distance (Hungarian matching). Report the distribution across runs, and show the lowest-error run’s court plot as an illustration, not as the result.

Dead cells. A cell is dead if it wins zero shots, or under about 0.1% of them, in the final pass. Usually it was seeded somewhere empty (the modern mid-range, behind the backboard) or got crowded out. Fixes, in roughly my order of preference:

Move dead cells. After each pass, move every dead cell to a random shot from the cell with the highest total error, which splits the worst-fit cluster. This keeps the same objective as plain competitive learning, so equilibrium positions stay interpretable.  
Leaky learning (Rumelhart & Zipser). Losing cells also move toward each input, at a much smaller rate. It has a nice neural story for your course, but it pulls every cell toward the overall mean of the data, so reduce the leak to zero by the end of training.  
Conscience (DeSieno). Penalize cells that win often. It works, but it pushes every cell toward winning 1/k of the shots, which by design puts about a third of your cells at the rim. That’s the outcome you were worried about, now built in. Use it only early in training, then switch it off.

Whatever you choose, record the number of dead cells per run as a diagnostic, and report the number of live cells (effective k) when comparing k values. Leave dead cells out of the geometric analysis, since their positions are leftover seeds, not anything learned.
