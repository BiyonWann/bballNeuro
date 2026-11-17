# Chat log #7

---

## User

We train one network through 2000-01, 2008-09, 2015-16 and 2024-25 in order. Each season starts from the previous season’s weights, and the learning rate resets to 0.05 each time. Why reset the learning rate instead of letting it keep shrinking?

---

## Claude

Because each season is a separate learning problem. A schedule that keeps shrinking only makes sense when the data stays the same.

A decaying learning rate assumes one fixed target. You start big to move quickly, then shrink so the weights settle into a minimum of a single loss. Here the data changes at every season boundary. Shot distributions in 2000-01 look very different from 2015-16 or 2024-25: far more threes, far fewer long twos. If the rate kept decaying across all four seasons, the network would arrive at 2024-25 taking tiny steps. It would stay close to the solution it found for the earlier seasons and barely fit the new one. Training order would decide the final model more than the data would.

Resetting gives you a warm start without freezing the weights. Starting from the previous weights carries over what stays the same across eras, such as court geometry and the basic link between distance and make probability. Each season then starts in a sensible place instead of from scratch. Resetting to 0.05 gives the network enough plasticity to actually shift toward the new season’s pattern. This is the same idea behind warm restarts (as in SGDR) and fine-tuning on a new task.

It also keeps the seasons comparable. Each season gets the same optimization budget: same starting learning rate, same schedule, same number of steps. So any difference in how much the weights or predictions change between seasons comes from the data, not from where that season sat in a shrinking schedule. If you’re analyzing how the learned representation drifts from era to era, this is what makes that analysis fair.

The tradeoff is forgetting. A fresh high learning rate will partly overwrite what the network learned about earlier seasons. After 2024-25 it probably predicts 2000-01 shots worse than it did right after training on them. If you want to track how the game evolved, that’s fine, and arguably the point: the network follows the league forward in time. If you need one model that handles all four eras well, you’d want something different, such as mixing older seasons back into training (replay), using a smaller reset rate, or training jointly on all four seasons.

---

## User

The share of test shots won by cells beyond 20 ft goes 0.30, 0.34, 0.35, then 0.50, and the mid-range share drops to 0 by 2024-25. Walk me through what that says about how the network’s map of the court changed.

---

## Claude

Read it as a story about where the map spends its cells. I’m assuming “won” means a cell is the best match for a test shot, as in a self-organizing or competitive-learning map.

What the metric measures. The cells are prototypes that compete for shots, so their positions follow wherever shots are dense. “Share of test shots won by cells beyond 20 ft” tells you how much of the map’s resolution goes to long-range territory, judged by the shots it actually has to represent. When the share rises, the network has moved prototypes outward, or packed more of them out there, because that’s where the data pulled them.

Phase 1: gradual drift (0.30 → 0.34 → 0.35). From 2000-01 to 2015-16 the map barely reorganizes. The warm start keeps most cells near their old spots, and each reset learning rate only nudges them as the long-range share of shots creeps up. The court layout is essentially the same, just tilted a bit more toward distance. The 2008-09 → 2015-16 step is almost flat, so the data in those two seasons wasn’t different enough to force reallocation.

Phase 2: a large reorganization (0.35 → 0.50). Between 2015-16 and 2024-25, half the test shots go to long-range cells. That isn’t a nudge. A big block of cells has migrated outward and now tiles the area around the arc much more finely, which matches the league’s shift toward threes over that decade. The gap between those two seasons is a bit longer, but the change is far larger than in the earlier gaps, so the data shifted faster in that stretch.

The mid-range going to 0. This is the most telling number, and it can mean either of two things:

The cells left. Nothing still has its prototype in the 10–20 ft band. Those units were pulled out to the arc or in toward the rim. Mid-range shots haven’t disappeared from the data, but they’ve become too sparse to keep their own cells. The few remaining ones get captured by whichever neighbor is nearest, so they’re counted as long-range or paint shots. That can inflate the 0.50 a little.  
The cells stayed but died. The cells are still sitting in mid-range but never win a test shot. That’s a different finding: the map is wasting units instead of reallocating them.

Either way, the 2024-25 map no longer treats mid-range as a distinct region. The network’s picture of the court has collapsed into rim plus perimeter, with mid-range absorbed into those two.

One caveat before you write this up. Win share mixes two things: where the cells are, and where that season’s test shots fall. Part of the rise from 0.35 to 0.50 would show up just because the 2024-25 test set has more long shots, even if the map hadn’t moved. Two checks would separate the effects:

Look at the cells themselves. Count how many prototypes sit beyond 20 ft and in mid-range in each season’s map, ignoring test shots. Moving counts mean the map reorganized. Flat counts with a rising win share mean the shift is mostly in the data.  
Use a fixed test set. Run every season’s map on the same shots, such as one pooled set or the 2000-01 test set. Any change in win share then has to come from the map.

Checking for dead units in the 2024-25 map would also tell you which version of the mid-range story is true.

---

## User

The average cell moves about 1 ft into 2008-09, then 0.8 ft, then 2.6 ft into 2024-25. Does a bigger jump mean the league changed more in those years? Or could it be because 2015-16 to 2024-25 is a nine-year gap while the others are about seven?

How is training one network through the seasons different from training a fresh one on each season? What can the sequential version show that fresh ones can’t?

---

## Claude

Does the bigger jump mean the league changed more?

Mostly yes, and the gap length can’t explain it. The network has no idea how many years separate two seasons. It only sees two datasets and gets the same training budget on each, thanks to the learning-rate reset. Elapsed time only matters indirectly, because more years give the league more time to change. So cell movement measures how different one season’s shot distribution is from the previous one’s.

Even if you credit the longer gap, the numbers don’t fit. Per year of gap:

2000-01 → 2008-09: 1.0 ft / 8 yrs ≈ 0.13 ft/yr  
2008-09 → 2015-16: 0.8 ft / 7 yrs ≈ 0.11 ft/yr  
2015-16 → 2024-25: 2.6 ft / 9 yrs ≈ 0.29 ft/yr

The last gap is about 1.3 times longer, but the movement is about 3 times larger. Per year, the rate more than doubles. That fits the win-share numbers: the league shifted faster after 2015-16.

Three things to check before you write “the league changed more”:

The noise floor. Cells move even when the data doesn’t change, because of sampling and shuffling and the reset learning rate kicking them around. Measure it by training on another resample of 2015-16 starting from the 2015-16 weights, then seeing how far the cells move. If that comes out around 0.6 ft, the first two jumps are mostly noise and the real change is concentrated in the last gap. That makes the story stronger, but you need the number to say it.  
Average vs. distribution. A 2.6 ft average could mean every cell moved a bit, or a handful of mid-range cells moved 10+ ft while the rest stayed put. Plot the per-cell displacements, or at least report the median alongside the mean. Given the mid-range collapse, I’d expect the second pattern.  
Endpoints only. Displacement compares the start and end of each gap. If the league drifted one way and came partly back within a gap, you’d never see it. With four seasons that’s a limitation to state, not something you can fix.  
Sequential vs. fresh networks

The difference is whether cell identity carries over from season to season.

With fresh networks, each season starts from its own random initialization. Cell 17 in the 2008-09 map has no relationship to cell 17 in the 2015-16 map. You can still compare the maps as whole distributions: win shares, cell density by zone, coverage. But you can’t ask how far a given cell moved. Some of the differences between maps also come from the initialization rather than the data.

With sequential training, cell 17 is the same unit throughout, so you get things fresh networks can’t give you:

Displacement at all. Your 1.0 / 0.8 / 2.6 ft numbers only exist because of sequential training. Fresh maps have nothing to subtract.  
Where the cells came from. You can trace individual trajectories and see whether the cells that ended up at the arc were the old mid-range cells, or whether arc coverage came from cells further in shifting outward. You can also see whether the departing mid-range cells split between rim and perimeter. That tells you how the map reallocated its resolution, not just that it did.  
Gradual vs. abrupt. You can tell whether a cell moved steadily across seasons or sat still and then jumped in the last one.  
History. Compare the sequential 2024-25 map with a fresh 2024-25 map. If they match, history didn’t matter and the sequential map simply tracks the data. If the sequential one has leftover cells in mid-range, dead units, or a different topology, the earlier seasons left a trace. That’s a finding in itself.

Fresh networks still earn a place as a control. A fresh map shows what a season looks like with no history, and running several seeds per season gives you the noise floor from point 1. Your strongest version probably uses both: the sequential run for trajectories and displacement, fresh runs to show which differences come from the data versus the model’s history
