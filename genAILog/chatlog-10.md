# Chat log #10

---

## User

Our competitive learning network trains on 20,000 shot locations per season and is never told where the 3-point line is.With 6 cells, the cells beyond 20 ft win shots that are 87% threes in 2024-25. With 12 cells that goes up to 95%, and accuracy to 0.980. Why does adding cells make the arc cells purer?

---

## Claude

Adding cells makes the arc cells purer because each cell can only draw straight borders, and the 3-point line is curved and has corners. More cells means more, shorter border segments, so they can follow the line more closely. Three things work together here.

1. The border can only be made of straight pieces. In winner-take-all competitive learning, each shot goes to its nearest prototype. That splits the court into Voronoi cells, and every cell is a convex polygon. The border between two cells is the straight perpendicular bisector between their prototypes. The 3-point region isn’t convex: it’s the outside of a circle, plus two straight corner strips. So the network’s “3-point zone” is really a polyline drawn around a curve. Wherever that polyline cuts inside the arc, the arc cell picks up long twos.

The size of that error shrinks fast as the cells get narrower. A straight segment across an arc of radius R that spans an angle θ sags inward by about R(1 − cos(θ/2)). The arc above the break spans roughly 136° (it meets the corner lines about 22° above the baseline):

2 arc cells: each covers about 68°, so the straight border can sag about 4 ft inside the line at the middle of each wedge.  
4 arc cells: each covers about 34°, so the sag drops to about 1 ft.

Halving the wedge width cuts the sag about 4x. That’s why going from 6 to 12 cells helps more than you might expect.

2. Prototypes stop getting pulled toward the middle. With 6 cells, an outer prototype often has to cover both the 3-point band and some 18–22 ft shots. Its position is a compromise between the two, so it sits inward, and the border with the inner cells moves inward too. With 12 cells, a separate mid-range prototype can take those long twos. The arc prototypes then move out to where threes actually cluster, about 24–26 ft, and the border moves out toward the real line.

3. The corners get their own cells. A corner three is 22 ft from the basket, but a 22 ft shot from the wing is a two. Distance from the basket alone can’t separate those. With only a few cells, one big outer cell can’t encode “22 ft counts in the corner but not on the wing.” With 12, the dense corner-three clusters can each win their own prototype, and that cell’s borders follow the corner geometry.

Why 6 cells already gets 87%. Modern shot charts have very few shots just inside the line (teams avoid long twos), so there’s a gap in shot density there. Competitive learning puts prototypes where shots are dense, so the borders tend to fall in that sparse gap. The sag regions from point 1 hold relatively few shots. More cells shrink those regions further.

Expect diminishing returns. You’re probably close to the ceiling now. The last ~2% likely comes from shots with a toe on the line and from noise in the location data. No placement of borders can separate those, so going past 12 cells should flatten out rather than approach 100%.

A check for your write-up: plot the misclassified shots at k = 6 and k = 12. If this explanation is right, the k = 6 errors should form crescents in the middle of each wedge and near where the corners meet the arc, and they should shrink sharply at k = 12. A plot of purity against k (6, 8, 12, 16, 24) would also show whether it saturates.

---

## User

In 2000-01 the network only pulls ahead of always guessing “2” once it has 12 cells (0.853 against 0.830). Is that the line starting to show up, or could more cells beat the baseline just by splitting the court into smaller pieces?

Our random baseline puts cells at random shots with no training. In 2000-01 its accuracy (0.847) actually beats the trained network, but its placement error is much worse (6.25 ft against 4.32 ft) and it varies a lot between seeds. How can an untrained map guess better, and why shouldn’t I read that as random being better?

n 2024-25 the network beats k-means on accuracy (0.944 against 0.917) and on arc purity (0.87 against 0.83), even though k-means is the standard method for finding cluster centers. Why would learning one shot at a time end up better at finding the 3-point line than k-means?

---

## Claude

1. 2000-01: the line, or just more pieces?

Some of that gain is guaranteed no matter where the cells go. Accuracy is presumably computed by labeling each cell with the majority class of the shots it wins. If you split a cell in two, the halves can’t do worse on those shots than the original cell did. So on the shots used for labeling, more cells never lowers accuracy, and any arrangement fine enough will eventually beat always guessing “2.”

In 2000-01, about 17% of shots are threes (1 − 0.830). A cell only gets labeled “3” if more than half of its shots are threes. With 6 cells, every cell probably still has a majority of twos. That makes the network identical to the always-“2” baseline, which fits your 0.830. At 12 cells, the first cell (or two) crosses 50% threes, and accuracy rises above the baseline. That step tells you a three-heavy cell finally exists. It doesn’t tell you the cell found the line rather than a three-heavy patch.

To tell the two apart:

Look at the cells labeled “3.” Do their prototypes sit at about 23–26 ft? Do their borders lie close to the arc, or do they cover one small three-heavy patch, like a corner?  
Compare against random placement at the same k. That shows how much splitting alone buys. The network should beat it by a margin larger than the spread across random seeds.  
Label the cells on one half of the shots and score on the other half. The splitting effect only holds on the shots used to label the cells. If the gain survives on held-out shots, it’s real structure.

Expect a weak signal either way. In 2000-01 there was no empty zone just inside the line. Mid-range shots were everywhere, and the threes sat in a thin band right next to them.

2. Why random placement can score higher, and why that doesn’t make it better

The network never tries to classify anything. It moves prototypes to reduce placement (quantization) error: the average distance from each shot to its winning prototype. Accuracy is a side readout. The two goals can disagree, and 2000-01 is where they do.

Training moves each prototype to the average position of the shots it wins. If one cell covers both 18–22 ft twos and 24 ft threes, its prototype settles between them, at about 21–22 ft. That’s the right spot for low placement error. But it leaves a cell straddling the line with a majority of twos, so it adds nothing to accuracy.

Random placement puts prototypes on actual shots. If a seed happens to drop one at 24 ft, that cell is centered on the 3-point band, and it may come out majority-threes. So a lucky seed can score higher on accuracy even though its placement is worse overall (6.25 ft vs 4.32 ft).

Reasons not to conclude random is better:

Spread across seeds. One number isn’t a result. Run 20–50 seeds and report the mean ± SD, plus how often random beats the trained network. I’d guess the mean is close to the network’s, with a wide spread.  
It’s luck, not a method. Nothing about random placement aims for the arc. A good seed is one where the dice happened to land on the band.  
Placement error is the score the network optimizes. Random loses that comparison by almost 2 ft, which is a large gap.

A good way to write it up: in a sparse-three era, getting accuracy out of the map depends more on where the cell borders happen to fall than on how well prototypes fit the shots. The trained network fits the shots better and doesn’t happen to draw a border at the line.

3. Why the online network might beat k-means in 2024-25

Plain competitive learning and k-means are chasing the same goal: low placement error. Updating one shot at a time doesn’t give the network any extra knowledge of the line. So the gap has to come from where each method ends up, or from something your network does that plain k-means doesn’t. Possible explanations, roughly from most to least likely:

Different stopping points. k-means stops at whichever local optimum its starting positions lead to. The online updates are noisy, and with a decaying learning rate that noise can shake prototypes out of a poor arrangement, like merging the arc with long twos, before they settle. If you ran k-means with a single initialization, try k-means++ with many restarts (n_init of 10 or more).  
A different goal. If your network has a conscience or frequency-sensitive term, a neighborhood function, or a step that revives dead units, it isn’t minimizing the same thing as k-means. Mechanisms that push cells toward equal win counts move prototypes out of the very dense rim area and into the 3-point band, which helps arc purity.  
The prototypes aren’t exactly averages. With a learning rate that never reaches zero, each prototype is biased toward the last shots it saw. If shots aren’t shuffled, that means late-season shots.  
Seed noise. 0.944 vs 0.917 might be inside the spread across seeds for either method. Run both across many seeds before treating it as a difference.

The check that sorts these out is to compare placement error for both methods. If the network has lower placement error, it found a better arrangement of the same goal (the first explanation). If k-means has lower placement error but lower accuracy, the network is optimizing something different, or it’s luck (the other three). Either way, the 2024-25 line shows up for both methods for the same reason: the line shaped where players shoot. Neither method is looking for it.
