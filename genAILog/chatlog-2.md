# Chat log #2

---

## User

We start each cell at a random shot. A lot of shots are at the rim, so I’m worried many cells start there. Can you explain what a dead cell is and why random starts could cause one?

---

## Claude

A dead cell is a cell that never or almost never wins a shot. Each shot gets assigned to its nearest cell, so a cell that is never nearest gets no data. It never moves and ends up as a useless point on the court. You asked for K cells but only get the benefit of K minus the dead ones.

Why starting at random shots causes this

Picking a random shot to start each cell sounds safe, because every cell begins on real data. The catch is that your starting points follow the shot density. If about 35% of shots are at the rim, about 35% of your cells start in that one small area. Three things can go wrong from there:

Exact duplicates. Rim shots often share the same coordinates, especially if locations are binned or rounded. If two cells start at the same spot, every shot is equally close to both. argmin breaks ties by taking the lower index, so the second cell loses every time, forever. With many identical rim shots, this is the most likely way to get a dead cell.  
Crowding. Even without exact duplicates, several cells packed into a few square feet split the rim shots between them. As the cells around one of them shift toward their own shots, that cell’s region can shrink until it wins nothing. In online winner-take-all learning this can’t recover: a cell that stops winning stops updating.  
Wasted coverage elsewhere. Every extra cell at the rim is a cell missing from midrange, the corner threes or the wings. So you get dead or redundant cells at the rim and too few cells everywhere else.  
How to check  
Count wins per cell after training, or after each epoch. Cells with zero or a tiny share of shots are dead.  
Check the starting positions for duplicates, for example with np.unique(init, axis=0). If you get fewer unique rows than cells, you already have dead cells before training starts.  
Fixes, roughly from easiest to most robust  
Sample from unique shot locations instead of from all shots. This removes the tie problem immediately.  
Use k-means++-style starts. Each new cell is picked with probability proportional to its squared distance from the cells already chosen, which spreads them across the court.  
Revive dead cells. Every so often, move any cell with zero wins onto the shot that is currently worst represented (the one farthest from its cell).  
Add a “conscience” (frequency-sensitive learning), which penalizes cells that win too often so the losers get a turn.  
Use a neighborhood update, as in a self-organizing map, if your model allows it. Neighbors of the winner move too, so no cell is left behind entirely.

If you tell me what training rule you use (batch k-means, online winner-take-all, or a SOM), I can say which of these fits best. I can also look at the code if it’s in one of your project folders.

---

## User

What does "leaky learning" do to fix that, and why do the losing cells use a smaller rate? Is there a downside if the leak rate is too big?

---

## Claude

What leaky learning does

In plain competitive learning, only the winning cell moves toward each shot:

Winner: w_win ← w_win + η_w · (x − w_win)  
Losers: don’t move

Leaky learning also moves every losing cell, but by a much smaller amount:

Losers: w_j ← w_j + η_l · (x − w_j), with η_l ≪ η_w

A dead cell used to sit still forever. Now every shot gives it a small pull toward the data, so it slowly drifts toward where shots are. Eventually it gets close enough to some group of shots to win one. From then on it gets the full winner rate and becomes a normal cell. Leaky learning doesn’t stop cells from losing; it just means losing no longer leaves a cell stuck for good.

Why losers use a smaller rate

The winner update is what makes cells specialize. Each cell moves toward the shots it wins, so it settles at the center of its own region. The loser update is a correction that pulls every cell toward all shots, which is the opposite of specializing.

If both rates were equal, every cell would move toward every shot by the same amount. All cells would end up at the same point, the average of all shots, and you’d have K copies of one cell. Keeping η_l small lets the specializing pull win out for active cells. For a dead cell, the leak is the only pull it gets, so even a small rate eventually brings it back.

The downside of a leak rate that’s too big

A cell’s settled position is roughly a weighted average of two things:

its own shots, weighted by η_w × (share of shots it wins)  
everyone else’s shots, weighted by η_l × (share it loses)

The catch is that a cell loses far more shots than it wins. Take a corner-three cell that wins 2% of shots, with η_l = 0.05 · η_w:

own weight: η_w × 0.02 = 0.02 η_w  
leak weight: 0.05 η_w × 0.98 ≈ 0.049 η_w

Most of the weight (about 70%) is on other cells’ shots, so the cell is pulled most of the way out of the corner. A leak that looks small (“5% of the winner rate”) can still dominate cells that cover sparse areas.

With your data this gets worse. The average of all shots is pulled toward the rim, because that’s where most shots are. A big leak therefore:

Pulls sparse-region cells inward. Corner and wing cells drift toward the paint, and you lose coverage exactly where shots are rare.  
Sends revived cells into the crowd. A dead cell brought back by the leak drifts toward the most crowded area. That’s the rim, where you already have too many cells, so it may just become a near-duplicate again.  
Keeps cells from settling. Every shot moves every cell a little, so positions keep jittering.  
At the extreme, collapses all cells to one point, as described above.  
Practical settings  
Keep the leak small relative to your sparsest region. For the leak not to dominate, η_l / η_w should be well below the share of shots your sparsest useful cell wins. With cells winning about 2%, something like 0.001–0.005 is in a reasonable range; 0.05 is too much.  
Decay the leak over time. A larger leak early on brings dead cells back. Shrinking it toward zero later lets cells settle at their true centers.  
Or leak only to dead cells. Apply η_l only to cells with zero (or very few) wins in the last epoch, so active cells aren’t biased at all. Combined with starting from unique shot locations, this fixes dead cells without moving the cells that already work.

---

## User

If I see a dead cell in one of my seeds, how do I tell whether it’s a bug in my code or just how the model behaves?

---

## Claude

The quickest way to tell is to look at when the cell died and whether it behaves the way the update rules say it should. Model behavior has a cause you can point to, usually crowding at the rim. A bug usually shows up as something your equations say can’t happen.

Step 1: Rerun the same seed

Run that seed twice and check you get the same dead cell at the same index with the same win counts. If the result changes between runs with a fixed seed, something isn’t seeded (shuffle order, init, a second random generator). Fix that first; nothing else is reliable until you do.

Step 2: Find out when it died

Log each cell’s win count per epoch, or at least its wins in epoch 1 versus the last epoch.

Zero wins from the very first step: almost always a duplicate or near-duplicate starting point. Check whether init[dead] equals another cell’s starting position. That’s expected model behavior given random-shot starts: a design flaw, not a coding bug.  
Won early, then dropped to zero: it was crowded out. Its neighbors should be nearby in a dense area like the rim or paint. That’s also model behavior.  
Zero wins, and its position is nowhere near any shot: suspicious. A cell that started on a shot shouldn’t end up far from the data. Go to step 3.  
Step 3: Check that the dead cell moves the way it should

With leaky learning on, a dead cell must still move a little every step: η_l · (x − w).

It moves slowly toward the data: the leak works; it just hasn’t revived yet. That’s model behavior. A larger or longer leak would bring it back.  
Its position never changes at all: that’s a bug. Common causes:  
the leak is computed but never written back, e.g. updating a copy instead of the array  
the loser mask excludes the wrong cells, or includes none  
the leak rate is 0 because of integer division or a config value that never gets passed in  
Step 4: Check the winner with brute force

For a few shots, find the winner the slow, obvious way and compare it with what your code picks:

for x in shots[:20]:  
    d = [np.sum((x - w)**2) for w in weights]   # loop, no broadcasting  
    assert int(np.argmin(d)) == winner_from_your_code(x)

A mismatch points to bugs like argmin over the wrong axis, broadcasting (N,1,2) against (K,2) incorrectly, or x and y swapped. Also check that the starting positions and the training data use the same coordinates. Starting cells from raw court coordinates while training on normalized ones (or feet vs. inches) puts cells off the court, and they never win.

Also check how you count wins. np.bincount(winners) without minlength=K silently drops trailing cells that won nothing, which shifts or hides dead cells.

Step 5: Compare across seeds

Run 10–20 seeds and record which cell indices die.

Pattern	Likely cause  
Different indices die in different seeds, usually in crowded areas	Model behavior  
Number of dead cells drops when you use unique starting shots or k-means++	Model behavior (the fix works)  
The same index dies in every seed (often cell 0 or cell K−1)	Bug: off-by-one, a cell skipped in an update loop, or a tie-break that always loses  
Number of dead cells doesn’t change no matter how you start the cells	Bug: the starting positions probably aren’t being used  
Step 6: Test cases where you know the answer

Two quick synthetic checks tell you whether the code does what you think:

Should give no dead cells: K well-separated blobs with one starting cell in each. Every cell should win its blob. If any dies, it’s a bug.  
Should give exactly one dead cell: two cells with identical starting positions, no leak. Exactly one of them should die. With a small leak turned on, it should slowly revive.

If your code passes both and the dead cell from your seed matches step 2 (a duplicate start or crowding at the rim), it’s how the model behaves, not a bug. Fix it with the starting-position and leak changes from before, not by debugging.
