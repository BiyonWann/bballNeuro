# Discovering the 3-Point Line with Competitive Learning

PSYC 40 / COSC 16 / COGS 21, Fall 2026 (Prof. Richard Granger)
Programming project. Biyon Wanninayake

---

## 1. Overview

The most important line on a basketball court is the 3-point line, and players have historically adapted to it. Over the past two decades teams have judged long two-point shots to be inefficient, so shots have shifted toward the basket or behind the 3-point line. Due to this, there is a clear separation in where shots are taken. This project asks whether a neural network that only sees where shots were taken can discover the 3-point line on its own. A competitive learning network (Rumelhart & Zipser, 1985) is trained on the court locations of NBA field-goal attempts and is never told where any line is. The same network is then trained season by season from 2000-01 to 2024-25 to see how its learned map of the court shifts as the league moved toward the 3-point shot.

## 2. Objective

**Q1: Can competitive learning discover the 3-point line from shot locations alone?**

The network only receives shot locations. During training, each cell's weights converge to a point on the court called a "prototype," which marks the center of the shots that cell wins.

Players avoid the area just inside the 3-point line and gather just behind it, leaving a gap inside the line and a dense cluster of shots along the arc. If the network places cells in that cluster and nearly all of the shots those cells win are 3-point attempts, then the network has found the line using only where players chose to shoot.

We also conduct a more in-depth test of whether the network learns the shape of the line. The arc is 22 ft from the basket in the corners and 23.75 ft at the top, so if the network has learned the shape, corner cells should settle closer to the basket than cells at the top of the arc.

**Q2: How do the learned prototypes change across eras?**

Since the early 2000s the league has moved away from mid-range shots and toward the 3-point line. A single network is trained on the 2000-01, 2008-09, 2015-16 and 2024-25 seasons in order, starting each season from the previous season's weights. Because each update moves only the winning cell, the network can follow a changing distribution of shots without being retrained from scratch.

## 3. Data

**Source:** Shot-detail files from the public GitHub repository `shufinskiy/nba_data`, which mirror the NBA's official shot-chart data from stats.nba.com. Each row is one field-goal attempt, and a season has roughly 200,000 rows.

**Fields:** The model uses only `LOC_X` and `LOC_Y`, the shot location relative to the basket in tenths of a foot. `SHOT_TYPE` (2-point or 3-point) and `SHOT_ZONE_BASIC` (official zone) are used for evaluation, never shown to the model. `GAME_ID` is used to split training and test data by game.

**Seasons:** 2000-01, 2008-09, 2015-16 and 2024-25, which span the mid-range era through the modern 3-point era. If the 2024-25 file is not available, 2023-24 will be used instead.

**Preprocessing**

1. Remove backcourt heaves and rows with missing coordinates, leaving only the six official zones (Restricted Area, In The Paint (Non-RA), Mid-Range, Left Corner 3, Right Corner 3 and Above the Break 3). Row counts before and after filtering are reported.
2. Convert coordinates to feet with the basket at the origin, and compute each shot's distance and angle from the basket for the polar variant.
3. Use 20% of each season's games as a test set.
4. Sample around 20,000 training shots per season so each NBA era is compared equally.

Before training, the data is visualized with a shot heatmap on a half-court diagram for each season, a histogram of shot distance from the basket that shows the gap inside the 3-point line, and a bar chart of shot share by official zone across the four seasons.

## 4. Model: Competitive Learning

**Architecture.** The network has one layer of K output cells that receive two input lines, x₁ and x₂, with a weight wᵢⱼ on each connection. Each cell's weight vector (w₁ⱼ, w₂ⱼ) lives in the same space as the shots, so it can be plotted directly on the court.

**Inputs.** The main model uses court coordinates in feet. Averaging shots along a curved arc pulls a prototype slightly inside it, so a polar variant uses each shot's distance and angle from the basket instead. This lets each prototype's distance match the average distance of the shots it wins.

**Performance rule.** Shots are presented one at a time. The cell whose weight vector is closest to the shot wins and outputs yⱼ = 1, and every other cell outputs 0. Closeness is measured with Euclidean distance rather than the largest net input, netⱼ = Σᵢ wᵢⱼxᵢ, because net input only picks the closest cell when all weight vectors have the same length. Normalizing court coordinates to equal length would remove distance from the basket.

**Learning rule.** Only the winning cell updates its weights, using Δwᵢⱼ = k(xᵢ − wᵢⱼ), while all other cells keep their weights. Over training each cell settles near the center of the shots it wins, and this final position is its prototype. After training, every shot is assigned to its closest prototype to form the learned zones.

**Implementation.** The network is written from scratch in NumPy with K = 6, matching the number of official zones. Cells start at randomly chosen shots, and the learning rate starts at k = 0.05 and shrinks after each of 5 to 10 shuffled passes so the cells can settle. In sequential era training, k is reset each season so the cells can move again. Cells that never win a shot (dead cells) are tracked, and leaky learning will be added if any appear. Each run is repeated with 5 random seeds since results depend on where the cells start.

**Why this model.** Competitive learning forms categories without labels by placing cells where data is dense, so it responds directly to the gap the 3-point line creates. It also learns online, which lets one network follow the league season by season. Its winner-take-all step mirrors lateral inhibition, and its update strengthens the winner's connections in the same way as LTP (Hebb, 1949).

## 5. Metrics and Baselines

For Q1, cell allocation counts how many prototypes sit near the rim (under 8 ft), in the mid-range (8 to 20 ft) and beyond 20 ft. 3-point purity is the share of each cell's shots that are 3-point attempts, which should be near 100% for arc cells if the line was found. Line-recovery accuracy labels each cell 2-point or 3-point by the majority of its training shots and checks those labels on test shots. Long 2s just inside the line will count as errors, so accuracy should be higher in eras with fewer of them. The shape test compares the average distance of corner cells and top-of-the-arc cells in the polar variant.

As secondary measures, the Adjusted Rand Index (ARI) and a confusion matrix compare the learned zones with the six official zones, with ARI ranging from 0 for chance to 1 for an identical match. Quantization error (QE), the average distance from each test shot to its closest prototype, shows how well the prototypes represent unseen shots.

For the era comparison, the same cells carry across seasons, so change is measured directly as how far each prototype moves and how the share of shots won by arc cells versus mid-range cells shifts.

K-means finds K centers using all shots at once rather than one at a time, so the network's prototypes landing near its centers confirms the code is correct. Random prototypes with no learning show how much training improves each metric.

## 6. Experiments

1. Train the network on the 2000-01 and 2024-25 seasons separately with 5 seeds each. The prototypes and their regions are drawn on a half-court to show where the cells settle.
2. Measure how well each network found the 3-point line using the Q1 metrics, then compare the results with both baselines and the six official zones. Comparing the two seasons shows whether the line is easier to find in the modern era.
3. Train one network through all four seasons in order, keeping its weights from one season to the next. Plotting each prototype's path visualizes how the network's map of the court shifts as shots move from the mid-range to the 3-point line.
4. Repeat steps 1 and 2 with polar inputs to test whether corner cells settle closer to the basket than cells at the top of the arc.

## 7. Limitations and Possible Improvements

The main limitation is density bias, in which cells crowd toward common shot locations and regions with few shots receive few prototypes even when they matter strategically. Because the network follows density, it finds where shooters gather rather than the line itself. Its boundaries are also straight with Cartesian inputs and cannot follow the curved arc, which the polar variant only partly fixes. Finally, results depend on where the cells start. Starting at random shots biases cells toward the rim, which is why each run is repeated with 5 seeds.

## 8. Deliverable

A Jupyter notebook in a GitHub repository that also runs in Google Colab. The README explains how to download the data and run the notebook. `requirements.txt` lists numpy, pandas and matplotlib, with scikit-learn used only for the k-means baseline and ARI.

## References

- Granger, R. (2026). ANN summary notes. PSYC 40 course materials.
- Hebb, D. (1949). *The Organization of Behavior.* New York: Wiley.
- Rumelhart, D. E., & Zipser, D. (1985). Feature discovery by competitive learning. *Cognitive Science, 9*, 75–112.
- Shufinskiy, V. *nba_data.* GitHub. https://github.com/shufinskiy/nba_data
