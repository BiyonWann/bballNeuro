# Chat log #1

---

## User

I’m building a competitive learning network for my PSYC 40 project. It learns from NBA shot locations to see if it can find the 3-point line on its own. Before I look at the code, can you explain competitive learning to me simply? What does a cell actually represent, and what happens when one shot comes in?

---

## Claude

Competitive learning in one sentence

You have a small group of cells that compete to “own” each input. For every shot, the cell that best matches it wins and moves a little toward it. The other cells stay where they are. Over many shots, each cell settles over its own cluster of the data.

What a cell represents

A cell is just a set of weights, and the weights have the same shape as the input.

If each shot is an (x, y) court location, each cell also has an (x, y) weight. You can treat that as a point on the court: the cell’s current guess at the center of a group of shots.

So with 4 cells you have 4 dots on the court. They start at random spots, and training moves them. By the end, each cell stands for one region of the court, a “kind of shot.” Every location on the court belongs to whichever cell is closest, so the court gets divided into territories.

What happens when one shot comes in

Say a corner three arrives at (22, 3).

Compare. Each cell measures how close its weights are to the shot. This is usually the distance between the two points, or a dot product if the inputs are normalized.  
Compete. The closest cell wins. This is called winner-take-all, and only one cell responds.  
Learn (winner only). The winner moves a fraction of the way toward the shot:  
w_new = w_old + η · (shot − w_old)

η is the learning rate. With η = 0.1 the winner moves 10% of the way toward that corner three. The losing cells don’t change.

Repeat for the next shot.

Each cell becomes a running average of the shots it wins, and it specializes. A cell that wins corner threes keeps getting pulled toward the corner, so it keeps winning them. Nobody tells the network what the categories are. They come from the data.

How this connects to the 3-point line

The network doesn’t know there’s a line. It only finds where shots pile up. NBA shot locations have a strong pattern: lots of shots at the rim, lots just behind the arc, and relatively few in the long midrange. Competitive learning tends to put cells on those dense areas. If the network “finds” the 3-point line, that will most likely show up as a boundary between cell territories that follows the arc, or as several cells lined up just outside it.

That’s also the right way to frame your result. The network discovers the line because players change their behavior around it. It isn’t learning the rule. That’s a nice point for a psych project: structure in the world shows up as structure in behavior, and an unsupervised learner can pick it up.

Things to look for in the code  
Number of cells. With too few, the arc gets merged into bigger regions. With more, the shape can emerge.  
Normalization. Some textbook versions, like Rumelhart & Zipser’s, require each cell’s weights to sum to 1. That changes what “closest” means, so check which version you have.  
Dead cells. A cell that starts far from all the data may never win, so it never learns. This is a common failure mode.  
Input features. Raw (x, y) versus distance from the hoop. With distance only, the problem becomes one-dimensional, and the line would show up as a cutoff around 22–23.75 ft.

When you’re ready, share the code and I’ll walk through how it does each of these steps.

---

## User

We’re using Euclidean distance instead. Can you show me with a small example why net input would pick the wrong cell for court coordinates?

---

## Claude

Setup

Put the hoop at the origin, in feet. Each shot and each cell is an (x, y) point. Take two cells and one shot:

Cell A = (1, 2), sitting near the rim  
Cell B = (22, 10), sitting out by a wing three  
Shot = (2, 3), a short shot near the rim

A is clearly the right cell for this shot.

Net input (dot product) picks B  
Cell	Net input w·x	Euclidean distance  
A	1·2 + 2·3 = 8	√(1² + 1²) ≈ 1.4  
B	22·2 + 10·3 = 74	√(20² + 7²) ≈ 21.2

Net input says B wins, since 74 is far bigger than 8. Euclidean distance says A wins by a wide margin. The shot is 1.4 ft from A and 21 ft from B, so the dot product gets it badly wrong.

Why this happens

The dot product is w·x = |w| |x| cos θ. It measures two things at once: how well the directions line up, and how long the weight vector is. A cell far from the origin has a big |w|, so it gets a large score for almost any shot, however far away it is.

The two measures are connected by this identity:

|w − x|² = |w|² − 2(w·x) + |x|²

|x|² is the same for every cell, so it doesn’t affect who wins. That leaves |w|² − 2(w·x). Net input only agrees with Euclidean distance if every cell has the same |w|. This is why textbook versions normalize the weights, such as Rumelhart & Zipser’s weights summing to 1. Court coordinates have no natural normalization. Cells near the 3-point line have large |w| and cells at the rim have small |w|, so net input would let the outer cells take over.

You can check the identity on this example. For A, 5 − 16 + 13 = 2, which is 1.4². For B, 584 − 148 + 13 = 449, which is 21.2².

What it would do to your results

The cell with the biggest weights would win shots across the whole court. It would then move toward them, and the other cells would rarely win and barely train. You’d see dead cells and one huge territory, with no arc.

What to use in the code

With Euclidean distance, the winner is the cell with the smallest |w − x|². You can skip the square root, because it doesn’t change which cell is smallest. The update rule stays the same, w += η (x − w) for the winner only. Normalization stops being necessary, and you can use raw court coordinates directly.

---

## User

So the update is w = w + k(x − w). If k is 0.5 the cell moves halfway to the shot. Why do we shrink k every epoch instead of keeping it fixed? What would the prototypes look like at the end if we never shrank it?

---

## Claude

Short answer

Two things pull in opposite directions. A big k lets cells find their territory quickly, and a small k lets them stop bouncing around once they have it. Shrinking k gives you both, with the big steps early and the small ones later.

What a fixed k does to a prototype

Unroll the update. Each new shot gets weight k, the one before it gets k(1−k), then k(1−k)², and so on:

w ≈ k·x_last + k(1−k)·x_prev + k(1−k)²·x_prevprev + ...

With k = 0.5, the last shot is 50% of the cell’s position, the one before it is 25%, then 12.5%. The cell is essentially remembering its last 2 or 3 shots. It never averages over the cluster. It keeps chasing whatever it won most recently.

If a cluster’s shots are scattered with standard deviation σ around the cluster’s center, a prototype with fixed k jitters around that center. The spread of its position is σ · √(k / (2−k)):

k	Jitter around true center  
0.5	≈ 0.58σ  
0.1	≈ 0.23σ  
0.01	≈ 0.07σ

So at k = 0.5, if shots in a region are spread about 3 ft, the prototype keeps wandering about 1.7 ft from where it should be. That doesn’t go away with more epochs.

What your final picture would look like  
Prototypes sit near the dense regions but never settle. Rerun training with a different shot order, or stop one epoch later, and every dot moves a little.  
Territory boundaries wobble. A boundary between two cells is halfway between their prototypes, so when the prototypes jitter the boundary does too. A boundary near the arc could shift by a foot or more from run to run.  
Cells on the border keep stealing from each other. A shot near a boundary can pull one cell over, which lets its neighbor win the next shot and get pulled back.

For your project this matters, because the arc is a precise feature. If the boundary’s position is noisy, you can’t tell whether a cell sits at 23.75 ft because of the line or because of the last few shots.

Why shrinking fixes it  
Early, large k: cells start at random spots, so they need big moves to reach the data and sort out who owns what.  
Late, small k: each shot nudges the cell only a little, so the position becomes an average over hundreds of shots and the noise washes out.

The ideal case is k = 1/n for the n-th shot a cell wins. The update then gives exactly the running mean of every shot it has won. Practical schedules like k₀ · decay^epoch or k₀ / (1 + epoch/τ) approximate this.

The risk on the other side

If you shrink too fast, cells freeze before they reach their clusters. A cell that hasn’t found its region by then stays stuck, and you get the dead-cell problem again. If your final prototypes look uneven, with a couple of cells clumped together or one stranded, check whether k has dropped near zero within the first few epochs.

---

## User

Here’s our train function: /Users/biyonwanni/Library/Application Support/Claude/scratch-workspaces/fcf4a7be-e7e2-4de2-b463-342de34dbf81/dce01e97-9ec6-4760-8dd9-85b2ad530562/scratch-2026-11-16-1e26c2/threept/network.py.Walk me through it line by line like I’m reading it for the first time. Point out where the winner-take-all part happens and where the learning rule happens.

---

## Claude

The path you gave points to an empty folder, but I found a threept/network.py in your neuroFinal project, under a worktree named network. I’m moving the session there so I can read it.

I read the file at threept/network.py in your neuroFinal project. The path you gave pointed to an empty folder, so this is the copy in the network worktree. If you meant a different copy, tell me. The path to the file is network.py.

Where the two key ideas live  
Winner-take-all: find_winner, lines 6-13, called from line 44 in train.  
Learning rule: lines 46-47, which is w + k * (x - w).  
Line by line

find_winner (6-13). This is the compete step.

It takes one shot (an (x, y) array) and weights, a matrix with one row per cell.  
Line 12: weights - shot subtracts the shot from every cell at once, giving each cell’s offset from the shot. np.linalg.norm(..., axis=1) turns each offset into a Euclidean distance, so you get one distance per cell.  
Line 13: np.argmin returns the index of the smallest distance. That cell wins, and the other cells get no say. This is winner-take-all. If two cells tie, the lower index wins.

init_weights (16-20). This sets where the cells start.

It picks n_cells different random shots from the data and uses them as the starting weights. The .copy() matters, because otherwise training would change your original data.  
Starting on real shots makes dead cells much less likely, since every cell begins in a spot where shots exist. This is a common fix for the dead-cell problem I mentioned earlier.

train setup (23-39).

The defaults are 6 cells, 8 epochs, a starting learning rate of 0.05, and a seed for repeatability.  
Lines 33-38 pick the starting weights. They use random shots by default, or the weights you pass in. They raise an error if the number of rows doesn’t match n_cells.  
Line 39 creates a random number generator, used for shuffling the shots.

The epoch loop (40-48).

Line 41: k = learning_rate / (1 + epoch). This is the shrinking k we talked about. It is 0.05 in epoch 0, 0.025 in epoch 1, about 0.017 in epoch 2, and 0.00625 in epoch 7. It shrinks once per epoch, not per shot.  
Line 42: this shuffles the order of the shots each epoch, so the cells don’t see them in the same sequence every time.  
Line 43: it loops over the shots one at a time.  
Line 44: compete. It finds the winning cell for this shot.  
Line 45: this is the optional leak step, covered below. With the default leak_rate=0.0 it does nothing.  
Lines 46-47: learn. w is the winner’s current position. The winner’s row is replaced with w + k * (x - w), which moves it a fraction k of the way toward the shot. This is the same update we’ve been discussing. Only that one row changes.  
Line 48: it returns the final prototypes.

leak_losers (51-61). This is an extra that plain competitive learning doesn’t have.

Line 56: if leak_k is 0, it returns immediately.  
Line 58: it builds a mask that is True for every cell except the winner.  
Line 61: it moves those losing cells toward the shot too, by leak_k * (x - w). Line 45 passes leak_rate * k, so the leak shrinks along with k.  
With a small leak, cells that never win still get pulled toward the data, which is another way to revive dead cells. At 0 you get pure winner-take-all.

The helpers (64-81). These analyze a trained network and don’t train anything.

assign finds the nearest cell for every shot at once, using broadcasting to build all shot-to-cell distances. These labels are what you’d color a court plot with.  
count_wins counts how many shots each cell owns.  
find_dead_cells returns the cells that own zero shots.  
Two things to check  
Integer data. init_weights copies rows straight from inputs, so the weights take the input’s dtype. If your coordinates are integers, such as raw NBA locations in tenths of a foot, the update on line 47 would be rounded to whole numbers on every step. Small moves could then round to nothing and the cells would barely learn. The initial_weights branch (line 38) already converts to float. Adding .astype(float) in init_weights would make both paths match. Print inputs.dtype to check.  
Aggressive shrinking. 1/(1+epoch) falls off quickly. By epoch 4 k is already at 0.01. With only 8 epochs, that’s fine if the cells have settled early, but check find_dead_cells on the result to be sure.
