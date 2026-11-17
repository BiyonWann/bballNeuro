# Chat log #6

---

## User

We compare the network to three baselines: always guessing the more common shot type, untrained prototypes placed at random shots, and k-means. Explain what each one rules out. What would it mean if the network beat only one of them?

---

## Claude

I’m assuming the metric works like this: each prototype gets labeled with the majority shot type of the shots it wins, and you score how well that label predicts the shot type. If your metric is different, the logic below still mostly holds.

What each baseline rules out

Majority class. This rules out class imbalance as the explanation. If 60% of shots are one type, 60% accuracy comes for free. Beating this only shows that court location tells you something about shot type. It’s the minimum bar, not evidence that the network learned anything.  
Untrained prototypes at random shots. This rules out “any partition of the court would do.” Because the prototypes start on real shots, they already follow where shots are dense. So this baseline is your network at epoch 0, and the gap between it and the trained network is the contribution of the learning rule itself. This is the most important control for a comp neuro project, since the claim you’re making is about learning. Average it over several random seeds and report the spread, because a single random draw can be lucky or unlucky.  
k-means. This rules out the worry that your learning rule finds a worse solution than the standard method. Online winner-take-all learning is essentially online k-means: each shot pulls its winning prototype toward it, so the prototypes drift toward their cluster means. That makes k-means a reference to match, not one to beat. Matching it means the biologically motivated rule finds about the same answer as the standard algorithm. Falling clearly short usually points to a known failure of competitive learning: dead units (prototypes that never win, so they never update) or a learning rate that doesn’t decay. Use the same K and the same evaluation for both.

So the three form a ladder. Is there any signal? Does learning add to it? Is the learned solution as good as the standard one?

If the network beats only one

In practice, “only one” can only mean the majority baseline. The other two cases point to a bug:

Beats majority, but not random prototypes (or k-means). The clusters carry information about shot type, but all of it comes from splitting the court into regions, and training added nothing. Either training didn’t work (learning rate, too few epochs, dead units), or the task is too easy. The second matters a lot if your shot types are defined by location, like 2PT vs 3PT. Then any fine enough partition gets close to the ceiling, and nothing can beat random prototypes by much. The fix is to sweep K. At small K, prototype placement matters and the comparison is informative. At large K, everything saturates. A plot of accuracy against K for all four methods makes this visible.  
Beats random prototypes but not majority. On the training data this can’t happen. Labeling each cluster by its own majority can never score below labeling everything with the global majority, because the global choice is just one of the options available to each cluster. If you see this, it’s either a bug or you’re scoring held-out data, where tiny clusters with noisy labels can overfit. Small clusters with unreliable labels also hint at the dead-unit or fragmented-cluster problem.  
Beats k-means only. That would mean it lost to the trivial baselines while beating the strongest one. That ordering doesn’t make sense, so check your evaluation code.

The result you’re hoping for is: clearly above majority, clearly above random prototypes at small K, and about equal to k-means. That supports the story “a simple Hebbian-style winner-take-all rule finds the same structure as the standard algorithm, and the structure comes from learning, not from the partition.”

---

## User

In 2000-01 the network’s accuracy is 0.83, exactly the same as always guessing “2”. In 2024-25 it’s 0.94 against 0.58. Is it fair to say the network found the line in one era and not the other? Or is the line just easier to see when more threes are taken?

In 2000-01 the network lands within 0.35 ft of the k-means centers, but in 2024-25 it’s about 4.2 ft away. In 2024-25 the network kept a cell around 10 ft, while k-means put more of its cells out on the arc. Why would two methods end in different places on the same shots, and does that mean something is wrong?

---

## Claude

1. Did the network find the line in one era and not the other?

No, that’s not a fair way to put it. The difference is mostly in the data, and partly in how the metric works.

An exact tie with the baseline is a clue. With majority-vote labeling, getting exactly 0.83 almost certainly means every cell’s majority label was “2.” That doesn’t mean no prototype sat near the arc. It means no cell had more threes than twos. In 2000-01, threes were only 17% of shots, and long twos were common right inside the line. So any cell near the arc probably mixed threes with long twos and still came out majority-2. The readout can only call a cell “3” if threes win the vote there, so it can’t show partial structure.

The network was never looking for the line. Competitive learning is unsupervised. It places prototypes where shots are dense, and the 3PT line is a label boundary it never sees. The line only shows up when the density structure happens to line up with it:

In 2024-25 it does. The midrange is hollowed out and shots pile up just beyond the arc, so a cell centered out there is mostly threes almost automatically.  
In 2000-01 shots taper off smoothly from the rim through the midrange and past the arc. No gap sits at the line, so no density-based method would split there.

So the better framing is: in 2024-25, where shots are taken lines up with the 3PT line, and in 2000-01 it doesn’t. That’s a finding about how shot selection changed. It isn’t evidence that the network succeeded in one era and failed in the other.

Checks to support that framing:

Run k-means in 2000-01. If it also scores 0.83, the result is about the data, not the network.  
Use a metric that doesn’t need a cell to flip to “3.” For example, normalized mutual information between cell and shot type, or use each cell’s 3PT fraction as a score and compute AUC. These can show that 2000-01 cells still sort shots by 3PT likelihood even though none of them wins the vote.  
Control for base rate. Subsample 2024-25 down to 17% threes and re-score, or report balanced accuracy in both eras.  
Sweep K. With more prototypes, a cell in 2000-01 may get small enough to be majority-3.  
2. Why do the network and k-means end up in different places?

Both methods minimize the same thing, but neither is guaranteed to find the best answer. Both are trying to minimize squared distance from each shot to its prototype. That objective has many local minima, and the two methods move through it differently:

k-means (batch): each step jumps every center straight to the mean of its current cell. A center can cross a sparse region in one step.  
Your network (online, decaying learning rate): a prototype moves only when it wins a shot, and by a shrinking amount each time.

So why would they agree in one era and not the other?

2000-01 has one obvious solution. Shot density is simple: a big rim cluster with a smooth falloff. Almost any starting point and update rule ends up in the same basin, so a 0.35 ft gap is basically the same answer.

2024-25 has several solutions that are nearly as good. Rim, corners, and arc are separate clusters with a sparse midrange between them. That’s exactly the setting with multiple competing local minima.

Why a cell could stay at 10 ft. That’s the sparse zone in 2024-25. A prototype that starts there rarely wins a shot, so it rarely moves. By the time it would drift out to the arc, the learning rate has decayed and it effectively freezes. k-means doesn’t have this problem, because its batch update moves a center all the way to its cell’s mean at once. A prototype stuck in a low-density region is a known weakness of online competitive learning.

Does this mean something is wrong? Not by itself. Three checks will tell you:

Compare total squared error (distortion). If the network’s is about the same as k-means’, you found a different but equally good solution, and the 4.2 ft gap just means the answer isn’t unique. If it’s clearly higher, the network got stuck.  
Run k-means with several random starts in 2024-25. If those runs also differ from each other by a few feet, then 4.2 ft is within the normal spread and isn’t specific to your network.  
Look at per-prototype distances, not just the average. Also check how you match prototypes to centers. Greedy matching can inflate the number, and the Hungarian algorithm gives the optimal pairing. If one prototype (the 10 ft one) accounts for most of the 4.2 ft, it’s a single stuck unit, not a disagreement about the whole layout. Plotting that prototype’s path over training will show whether it froze.

If it turns out to be stuck, the standard fixes are a slower learning-rate decay, more epochs, or a “conscience” mechanism that penalizes units that win too often so rarely-winning units get a chance. Frequency-sensitive competitive learning is the version of this you’ll find in the literature.

How the two connect

The same 2024-25 structure that lines up with the 3PT line (dense rim, empty midrange, dense arc) is also what creates competing solutions. So the era where the line is visible is also the era where the two methods disagree. That fits together, and it’s worth saying in the writeup. One last check: if k-means puts more cells on the arc, see whether it beats 0.94. If it does, the stuck 10 ft cell is costing you accuracy, not just distance.
