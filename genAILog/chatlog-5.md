# Chat log #5

---

## User

in bballNeuro, Our evaluate function scores a set of prototypes. It learns each cell’s type (mostly 2s or mostly 3s) from the training shots, then measures accuracy, quantization error, the adjusted Rand index and the shot shares on the test shots, and counts dead cells on the training shots. Why split it that way between train and test?

---

## Claude

The rule behind the split: anything the model learns comes from the training shots, and anything that measures how good the model is comes from the test shots. Dead cells are the exception because they describe the fitting itself, not how well the model generalizes.

Learning cell types on train. Labeling each cell as mostly-2s or mostly-3s is part of the model, just like the prototype positions. If you labeled cells using the test shots and then scored accuracy on those same shots, every cell’s label would be its majority vote over exactly the data being graded. Accuracy would be inflated by construction. Labeling on train and scoring on test asks the real question: does a cell’s learned type hold up for shots it has never seen?

Accuracy, quantization error, ARI and shot shares on test. Each of these asks how well the prototypes describe new data:

Quantization error: the prototypes were pulled toward the training shots, so error on train is optimistically low. Test error tells you whether the map captures the actual shape of the shot distribution or just memorized where the training shots landed.  
ARI: checks whether grouping shots by their nearest prototype (their best-matching unit) lines up with the real 2/3 split. That only counts as evidence of structure if it holds on held-out shots.  
Shot shares: say how the map divides unseen shots across cells and regions. That’s the number you’d report as “the map’s view of where shots come from.”

Dead cells on train. A dead cell is a prototype that no training shot maps to. That’s a failure of the training process: the map spent a unit on a region the data never filled, usually from a bad start or the neighborhood shrinking too quickly. Counting on the test set would mix that up with plain sampling noise. The test set is smaller, so some perfectly good cells will get zero test shots by chance, and you’d overcount. Counting on train also matches the labeling step: a cell with no training shots is exactly a cell that never got a type.

That last point raises an edge case to check in evaluate: what happens when a test shot’s nearest prototype is a cell that was dead in training? That cell has no learned type, so the code has to do something, like fall back to the overall majority class, use the nearest labeled cell, or drop the shot. Whatever it does should be deliberate and reported, because it affects test accuracy.

---

## User

What would go wrong if I learned the cell types and scored accuracy on the same shots?

---

## Claude

Explaining why the bias is optimistic and how it grows with more cells.

You’d get a number that is guaranteed to look good and can’t tell you anything, and it gets worse the more cells you have.

When the cell types come from the same shots you score, each cell is labeled with its own majority class, so every cell automatically gets at least half its shots right. Overall accuracy works out to

accuracy = Σ over cells of max(n₂, n₃) / N

where n₂ and n₃ are that cell’s counts of 2s and 3s. That has two consequences.

1. It can never fall below the baseline. Splitting shots into cells and taking each cell’s majority never does worse than taking the majority over all shots together. So same-data accuracy is always at least the “always predict 2” baseline, even if the prototypes were placed at random. Beating the baseline proves nothing.

2. It only goes up as cells get smaller. Splitting a cell into finer cells can’t lower the sum of majorities. In the extreme, with one shot per cell, every cell is pure and accuracy is 100%. So if you compared map sizes this way, the biggest grid would always win. The reason isn’t that it learned court geometry. It just has enough cells to memorize individual shots.

The deeper problem is that noise gets counted as signal. Say a cell holds 3 shots that happen to be 2, 2, 3. It gets labeled “2s” and scores 2/3 on itself, but that label is mostly chance. On new shots from the same spot, the real 2/3 mix might be close to even, and the label would do much worse. Scoring on the same data rewards the map for fitting those accidents. Held-out shots are the only way to tell “this region of the court really produces mostly 3s” apart from “this cell happened to catch a few 3s.”

So same-data accuracy measures how finely the map chops up the shots, not how well it understands them. That’s why held-out test shots have to be what you score on, and it matters most when you use accuracy to choose between grid sizes or training settings.

---

## User

On real data, 2024-25 gets 0.94 accuracy, but the cells beyond 20 ft are only about 0.87 threes. How can accuracy be higher than the purity of the arc cells? What does each number actually measure?

Why count dead cells on the training shots, when nearly everything else uses the test shots?

---

## Claude

These are two separate questions, so here they are in turn.

Why 0.94 accuracy can beat 0.87 arc purity

The two numbers average over different sets of shots.

Accuracy (0.94) is the share of all test shots whose cell’s learned type matches the shot’s real type. It’s a shot-weighted average of how well each cell’s label fits, taken over the whole court.  
Arc purity (0.87) is the share of 3s among shots in cells beyond 20 ft. It only covers the hardest part of the court.

Most cells are inside the arc: the paint, the short midrange and the floaters. Those shots are 2s almost without exception, so those cells are close to perfectly pure. Accuracy averages the arc’s 0.87 with those near-1.0 cells, so it lands in between.

The arc cells are impure because “beyond 20 ft” doesn’t line up with the three-point line. The line is 22 ft in the corners and 23.75 ft at the top of the key, so those cells also hold long 2s from 20 to 23.75 ft, plus shots taken with a foot on the line. A cell that straddles the line can’t do better than its majority share. That 0.87 is a limit of cell resolution, not of the labeling.

As a sanity check, take an arc share f with the inside cells around 0.99 pure:

0.94 ≈ f · 0.87 + (1 − f) · 0.99 → f ≈ 0.4

So roughly 40% of shots would land in arc cells, which is close to the share of NBA shots that are threes in 2024-25. If your actual numbers don’t come out near that, check that the purity and the accuracy were measured on the same split.

One thing this points to: almost all of the accuracy you lose comes from the cells that straddle the line. Per-cell accuracy, or accuracy split by distance band, would show where the map is failing much better than the single 0.94.

Dead cells on train

This is the same reasoning as before, with one way to make it concrete.

Every other metric asks how well the map generalizes. A dead-cell count asks whether training wasted any prototypes. Only the training shots can answer that, because they’re the data the prototypes were fit to.

On the test set, emptiness mostly reflects sample size. A cell that holds a share p of shots gets none of n test shots with probability (1 − p)ⁿ. On a big grid there are cells with small p, and with a smaller test set, some of them will come up empty just by chance even if the map is perfect. Those cells weren’t wasted. They just weren’t sampled.

Counting on train also lines up with the labeling step. A cell with no training shots never got a type, so “dead on train” and “unlabeled” are the same set of cells. That’s the set to watch when test shots land in them.

---

## User

If I could only report three numbers to answer “did the network find the 3-point line”, which three would you pick and why?

---

## Claude

I’d pick three numbers that each answer a different part of the question: where the learned boundary sits, whether it has the line’s shape, and whether it holds up on shots it hasn’t seen.

1. Boundary placement error, in feet. Label a fine grid of court points by the type of each point’s nearest prototype. Take the contour where the label flips from 2 to 3, and report the median distance from that contour to the real three-point line. This is the most direct answer to “did it find the line,” and the answer comes in court units (“the learned boundary is within 0.8 ft of the line”) instead of as a score. It depends only on the prototypes and their training labels, so it needs no test shots. It also isn’t dominated by easy paint shots the way accuracy is.

2. Corner-vs-top offset. Measure how far the learned boundary is from the hoop in the corners and at the top of the key, then take the difference. On the real line it’s 22 − 23.75 = −1.75 ft. This is what separates “found the three-point line” from “learned that far shots are threes.” A map that learned only a distance cutoff would draw a circle and get an offset near 0. Getting close to −1.75 means it recovered the line’s actual shape, which you can only get from the data’s geometry.

3. Test accuracy near the line. Score only the held-out shots within about 3 ft of the true line. Your overall 0.94 is padded by paint and midrange cells that anything would get right. This band is where the two shot types actually overlap, so it’s the honest test of whether the boundary generalizes. If you can add one sentence of context, give the best single-radius cutoff’s accuracy on the same band (with the radius chosen on train). Beating that cutoff in the corners is the clearest evidence the map learned something a distance rule can’t.

What I’d leave out:

Overall accuracy: dominated by easy cells.  
ARI: mostly says the same thing as accuracy when the labels are binary, and it’s harder to read.  
Quantization error: measures how well the map fits where shots are dense, not where the line is.  
Dead cells and shot shares: useful checks on training and on the data, but neither says anything about the line.

One practical point: numbers 2 and 3 are only as good as your coordinate setup. They need shot locations in feet measured from the hoop, not from the baseline or in raw image pixels. Otherwise the 22 and 23.75 ft references won’t line up with your data.
