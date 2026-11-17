# Chat log #9

---

## User

We train one network through 2000-01, 2008-09, 2015-16 and 2024-25 in order. Each season starts from the previous season’s weights, with the learning rate reset. The number of cells beyond 20 ft stays at 3 of 6 through 2015-16, then jumps to 5 of 6 in 2024-25 when the last mid-range cell moves out to the arc. Why would the change show up as one jump instead of a slow drift?

---

## Claude

The short answer is that a jump is what you’d expect from three things together: the statistic you’re counting, how winner-take-all learning settles, and how far apart your seasons are. A jump in the count doesn’t mean the change was abrupt in the data or in the weights.

1. “Cells beyond 20 ft” is a thresholded count. Each cell’s distance from the basket is continuous, but the count only changes when a cell crosses 20 ft. A mid-range cell could move from 15 to 19 ft across three seasons and the count would stay at 3/6 the whole time. So the jump may be partly a measurement artifact. Plot each cell’s radial distance per season before deciding anything moved suddenly.

2. 20 ft falls in a low-density gap. Competitive learning works a lot like online k-means. At convergence, each cell sits near the mean of the shots it wins, and those means land on clusters of shots: rim, short mid-range, corners, above the break. Long twos (about 18–23 ft) have become rare, so few cells ever settle at 20 ft. A cell is either clearly mid-range or clearly at the arc. Crossing the line means switching clusters, and that’s a discrete event.

3. Cells resist moving until their territory disappears. A cell only updates when it wins. As long as some mid-range shots remain, the mid-range cell keeps winning them and stays roughly where it is. The layout from last season stays stable while mid-range volume falls. Only when that cell’s share of wins gets small enough do the neighboring cells claim its region and push it out to the arc. Gradual change in the data, then a sudden rearrangement of the cells, is the usual pattern for this kind of system.

4. The warm start adds lag (hysteresis). Each season starts from the previous season’s weights, so the network keeps old structure longer than a fresh run would. A network trained from scratch on 2015-16 might already put 4 cells past 20 ft, and starting from 2008-09’s layout is what keeps it at 3. Resetting the learning rate then gives each season enough early step size to fully reorganize once the old layout stops working, so the change lands within a single season.

5. Your snapshots are spread unevenly. You go 2000 → 2008 → 2015 → 2024. The biggest part of the three-point shift happened inside that last nine-year gap. Any gradual drift between 2015-16 and 2024-25 isn’t in your data, so it can only show up as one step.

Ways to tell these apart:

Plot cell distance (and each cell’s share of wins) instead of the count. If distances creep toward 20 ft before the crossing, the jump is mostly the threshold. If they stay flat and then leap, it’s the reorganization in point 3.  
Train from scratch on each season with several random seeds. If the fresh runs differ from the warm-started run, the gap is the hysteresis from point 4.  
Add the seasons from 2016-17 through 2023-24, warm-starting through them in order. You’ll either find the one season where the layout flips or see it spread across several.  
Plot a histogram of shot distances for each season and check for the empty band around 20 ft from point 2.

One caveat for your write-up: a jump from 3 to 5 means two cells crossed, not one. Check whether the second cell was already near 20 ft in 2015-16. If it was, that supports the threshold explanation for at least one of the two crossings.

---

## User

The share of test shots won by the arc cells goes 0.30, 0.34, 0.35, 0.50, and the league’s actual three-point share goes 0.17, 0.22, 0.28, 0.42. Why is the arc-cell share always higher than the real three-point share? What else are those cells winning?

---

## Claude

The arc cells are almost certainly also winning long two-pointers, meaning mid-range shots taken just inside the line. They win them because of how winner-take-all decides who owns a shot.

Why: the cells’ borders aren’t the three-point line. Each shot goes to whichever cell is nearest. So the border between a mid-range cell at about 15 ft and an arc cell at about 25 ft falls roughly halfway between them, around 20 ft, not at the actual line (22 ft in the corners, 23.75 ft above the break). Any two-pointer between about 20 ft and the line goes to the arc cell.

Too few inside cells to cover every angle. If your inputs are court (x, y) coordinates, the cells split the court into wedge-shaped regions, not distance rings. With only 6 cells, one or two mid-range cells can’t cover the whole arc around the basket. A 17-footer from the wing may sit closer to the wing arc cell than to a mid-range cell parked near the top of the key. When that happens, the arc cell takes even mid-range shots.

Your own numbers fit this. Subtract the real three-point share from the arc-cell share:

Season	Arc-cell share	Real 3PT share	Excess  
2000-01	0.30	0.17	0.13  
2008-09	0.34	0.22	0.12  
2015-16	0.35	0.28	0.07  
2024-25	0.50	0.42	0.08

The excess shrinks over time, and the long two declined over the same period. As those shots disappear, there’s less for the arc cells to take. If the arc cells were picking up something unrelated, like layups or a bug, the excess wouldn’t follow that trend.

How to confirm it:

The shotdetail data has SHOT_TYPE (2PT/3PT) and SHOT_ZONE_BASIC columns. Cross-tabulate arc-cell wins against them. The 2PT share among arc-cell wins should roughly equal the excess in each season, and should fall mostly in the “Mid-Range” zone at 18–23 ft.  
Check the reverse as well. If inside cells win some three-pointers, the true number of long twos the arc cells take is a little higher than the excess shows.  
Plot each season’s cell borders over a court diagram with the real three-point line drawn on it. You’ll see where the borders cut inside the line.

One thing to rule out first: confirm the 20 ft cutoff uses the same units as your inputs. shotdetail’s LOC_X/LOC_Y are in tenths of a foot, so a 20 ft check needs to compare against 200 in those units.

---

## User

In 2008-09 the network’s accuracy exactly ties always guessing “2” (0.773). By 2015-16 it pulls ahead (0.766 against 0.714). What could have changed in the shots between those seasons that let the same network start beating the simple guess?

---

## Claude

Most of the “pulling ahead” comes from the baseline getting worse. The network’s own accuracy actually fell slightly, from 0.773 to 0.766. Always guessing “2” fell much more, from 0.773 to 0.714, because the three-point share on your test shots rose from 0.227 to 0.286. What changed between those seasons is the mix of shots: there were more threes, and fewer long twos near the arc.

The exact tie in 2008-09 is a clue, and how to read it depends on how the network turns a winning cell into a “2” or “3” prediction.

If each cell is labeled by the majority class of the training shots it wins

Then a tie to three decimals almost certainly means the network guessed “2” for every shot in 2008-09. No cell had more threes than twos, so every cell got the label “2” and the network was the baseline.

In 2008-09, about a third of all shots were won by the arc cells. But only about 23% of shots were threes, and some of those went to inside cells. Spread across three arc cells, each one could easily be mostly long twos.

By 2015-16, two things push an arc cell over 50% threes: three-point volume rose, and long twos were drying up, so there was less two-point traffic near the arc. Once one cell’s label flips to “3”, accuracy rises above the baseline. That would be another discrete jump, like the cell count in your first question.

To check, print how many “2” and “3” predictions the network made in 2008-09. If it made zero “3” predictions, this is the explanation.

If “arc cell” means a fixed rule that predicts “3” for any cell beyond 20 ft

Then the tie is a coincidence. Your arc-share numbers let you split the errors two ways: twos predicted as threes, and threes predicted as twos.

	2008-09	2015-16  
Real 3PT share	0.227	0.286  
Arc-cell share	0.34	0.35  
Total error (1 − accuracy)	0.227	0.234  
Twos predicted as threes	≈0.17	≈0.15  
Threes predicted as twos	≈0.06	≈0.08

These come from two equations: twos predicted as threes minus threes predicted as twos equals arc share minus real three-point share, and their sum equals total error. The arc-share values are rounded, so the splits are approximate.

Under this rule, the network’s overall error barely changed. The mistakes just shifted:

Fewer twos were predicted as threes, because long twos were declining.  
More threes were predicted as twos. Corner threes are the likely cause: corner-three volume grew a lot over this stretch, and a corner three at 22 ft can sit closer to a baseline mid-range cell than to any arc cell.

You can test the corner-three explanation by filtering the threes predicted as twos by SHOT_ZONE_BASIC and checking whether they’re mostly “Left Corner 3” and “Right Corner 3”.

Something to consider for the write-up: majority-class accuracy moves whenever the share of threes moves. Balanced accuracy, or recall on threes, would show whether the network actually separates twos from threes better, rather than just looking better because the baseline got worse.
