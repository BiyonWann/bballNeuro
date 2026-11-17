# Chat log #4

---

## User

in bballNeuro github repo, we had a bug where init_weights copied rows straight from the inputs with inputs[rows].copy(). If the inputs were whole numbers, the weights stayed whole numbers too, and then weights[winner] = w + k * (x - w) silently cut every update down to an integer. Walk me through why numpy does that on assignment instead of switching the array to float.

The fix was to return inputs[rows].astype(float). Why does that fix it, and why does it still give the caller a separate copy, so their array can’t be changed?

---

## Claude

Checking how the array copy and indexing behave.

NumPy truncates on assignment because an array’s dtype belongs to its memory buffer, and weights[winner] = ... writes into that buffer. It does not build a new array.

Why the assignment truncates

A NumPy array is a block of raw memory plus a header saying how to read it, such as “int64, 8 bytes per element.” A Python list is different: it holds references to separate objects, so each element can have its own type. An array can’t do that.  
Arithmetic makes a new array, so NumPy can choose the result type freely. w + k * (x - w) follows type promotion (int combined with float gives float64) and produces a new float64 array. That part of your code was correct.  
weights[winner] = expr calls __setitem__, which only fills slots in the existing buffer. To “switch to float,” NumPy would need a new buffer and would have to swap it out from under weights. That would break any views that share the old memory, and the type of an array would then depend on what you happened to write into it. So NumPy converts the right-hand side to the array’s dtype, and float to int conversion truncates toward zero.

I checked this against NumPy 2.5.3:

w = [5, 5, -5], x = [10, 0, 0], k = 0.1  
w + k*(x - w)      -> [ 5.5  4.5 -4.5]   float64  
w[:] = that        -> [ 5    4   -4  ]   int64

That output shows why the bug was hard to spot. Truncation toward zero isn’t just a loss of precision. It biases the updates:

A weight that should move away from zero, like 5 toward 10, gets stuck whenever the step is smaller than 1.  
A weight moving toward zero, like 5 toward 0, still moves.

So the weights drift toward zero instead of settling on the inputs, and nothing raises an error.

A contrast worth knowing

w += 0.5 does raise an error:

UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'

In-place operators use the stricter same_kind casting rule, while item assignment uses unsafe casting. Writing weights[winner] += k * (x - w) would have caught the bug immediately. The plain = assignment didn’t.

Why .astype(float) fixes it

astype builds a new array with a float64 buffer and converts each value into it. Since weights is float from the start, every later weights[winner] = ... writes floats into float slots, and there’s nothing to truncate.

Why the caller’s array is safe

Two separate things protect the caller here:

astype copies by default. Its copy=True default means it always returns a new array, even when the input is already float64 and no conversion is needed. Nothing you do to weights can reach inputs.  
inputs[rows] was probably already a copy. If rows is an integer array, like the output of rng.choice, that’s fancy indexing, which always copies. If rows were a slice like inputs[0:k], you’d get a view into the caller’s memory, and only the astype copy would protect them.

Your original .copy() was handling the same thing: it stopped the weights from sharing memory with the caller, but it kept the int dtype. .astype(float) covers both problems: it gives a separate buffer, and the buffer is the right type.

Since init_weights now depends on that copy, you could add assert weights.dtype.kind == 'f' or a test with integer inputs to catch the bug if it ever comes back. If you pass copy=False to astype someday to save memory, check np.shares_memory(weights, inputs) in a test, because then you’d only be protected when rows causes a copy.

---

## User

Our real inputs are decimals (feet, like 23.75), so the actual results were never wrong. If the bug had hit the real runs, what would the prototypes have looked like, and how could I have spotted it from the plots?

The test for it trains one cell on the shots (0, 0) and (1, 0) with a learning rate of 0.5 for one epoch, and checks the weight ends strictly between 0 and 1. Walk me through why every starting point and shot order gives 0.25, 0.5 or 0.75, while the buggy version can only give 0 or 1.

In general, where else can numpy quietly change a result because of the data type? What should I watch for when I review numpy code later?

---

## Claude

If the bug had hit the real runs, the prototypes would have ended up on whole-foot coordinates and been dragged toward wherever the court’s origin is, not sitting in the middle of their shot clusters. Your test catches the bug every time, but not quite the way you described: the buggy version never gives 1. It gives 0 for every starting point and shot order. Details on both, then the general list of dtype traps.

1. What buggy prototypes would have looked like

The bug only happens when the inputs are integer-typed. Your feet-as-decimals data is float64, so it was never at risk. It could come back if you load coordinates stored as integers. For example, pd.read_csv makes any column with only whole numbers into int64. If it did happen, three things would show:

Every prototype on whole-foot grid points. Initialization copies integer shots, and every update gets cut back to an integer.  
A pull toward the origin. As the earlier test showed (5 toward 10 gets stuck at 5, while 5 toward 0 drops to 4):  
A step away from 0 smaller than 1 ft is thrown away.  
Any step toward 0, however small, moves the prototype a full foot.  
Late in training the learning rate is small, so prototypes stop moving outward but keep stepping inward. Each one creeps until it reaches the shot in its cluster closest to the origin, one coordinate at a time.  
Where they end up depends on the origin. With the hoop at (0, 0), prototypes sit on the hoop-side and centerline-side edge of their clusters. A corner-three prototype would land at the cluster’s inner corner, and mirrored left and right clusters would both shift toward x = 0. With the origin at a baseline corner, everything leans toward that corner instead.

How you’d spot it from the plots:

Prototypes overlaid on the shot scatter sit at the edge of their clusters instead of the center, and all lean the same way.  
If you zoom in, the coordinates line up on an exact grid.  
In per-epoch snapshots, prototypes freeze early except for small moves inward.  
The quantization-error curve levels off higher than it should.

Plots only hint at this, though. print(weights.dtype) or np.all(weights == np.round(weights)) tells you directly.

2. Why the test gives 0.25, 0.5 or 0.75

The y-coordinate stays 0 throughout, so only x matters. Each update is a weighted average of the old weight and the shot:

w ← (1 − k)·w + k·x

With k = 0.5 and two updates, call the first shot’s x a and the second’s b:

w_final = 0.25·w₀ + 0.25·a + 0.5·b

init_weights copies a row of the inputs, so w₀ is 0 or 1. One epoch shows each shot once, so (a, b) is (0, 1) or (1, 0). That gives four cases:

start w₀	order of shots	result  
0	0 then 1	0.5  
0	1 then 0	0.25  
1	0 then 1	0.75  
1	1 then 0	0.5

The result is always strictly between 0 and 1. It’s a weighted average where the 0-shot and the 1-shot both get positive weight (0.25 or 0.5), and the weights add to 1. The 1-shot keeps the result above 0, and the 0-shot keeps it below 1.

Why the buggy version gives only 0

The “0 or 1” bound is right: each step’s float value is a weighted average inside [0, 1], and truncating it gives an integer, so the only candidates are 0 and 1. But 1 never survives:

Starting at 0: a step toward 1 gives 0.5, which truncates back to 0. A step toward 0 does nothing.  
Starting at 1: a step toward 1 does nothing. A step toward 0 gives 0.5, which truncates to 0, and from 0 the weight can’t climb back.  
Every epoch includes the 0-shot, so all four cases end at 0. I simulated all four and they matched the table above (float) and gave 0 every time (int).

This is a good property for the test: it fails every time, whatever the seed. The core reason is that the step size (0.5) is under 1, so truncation can only ever pull the weight down. If you want a stronger check, you could assert the result is one of {0.25, 0.5, 0.75}. That would also catch a wrong learning rate, not just the dtype.

3. Other places dtype quietly changes results

I checked each of these on NumPy 2.5.3.

Writing into an existing array keeps its dtype

a[i] = value truncates floats into int arrays. This is your bug.  
np.zeros_like, np.empty_like and np.full_like copy the dtype of their argument. np.full_like(int_array, 0.5) gives [0, 0]. It’s the same bug, set up a different way.  
a += b and out= keep a’s dtype. Float into int raises an error. But float32 += float64 is allowed and quietly stays float32.

Fixed-width integers wrap around with no warning

uint8: 200 + 100 gives 44, and 100 − 106 gives 250. Image data and small counters are where this usually shows up.  
int64 wraps too: np.array([2**62]) * 4 gives [0], with no warning even with warnings turned on. np.seterr only covers floating-point errors.  
Since NumPy 2, mixing a too-large Python int with a small-int array raises an error (uint8_array + 300 raises OverflowError). Arithmetic between two arrays still wraps silently.

Float precision

Python floats no longer upcast arrays (NumPy 2’s NEP 50 rules), so float32_array * 0.1 stays float32.  
Large sums and cumulative sums in float32 lose precision. Pass dtype=np.float64 to reductions when it matters.

Converting between types

astype(int) truncates toward zero instead of rounding: −1.7 becomes −1. Use np.rint(x).astype(int) if you mean rounding.  
np.array([1, 'a']) turns everything into strings (<U21), and mixing in None gives an object array.  
NaN has no integer equivalent. Converting it with astype gives a garbage value, and pandas turns int columns with missing values into float.

Masks stored as 0/1 integers instead of booleans

x[m] with an int array is fancy indexing, not filtering. [10, 20, 30][[1, 0, 1]] gives [20, 10, 20].  
~m is bitwise NOT, so it gives [-2, -1, -2], not the inverted mask.  
Checklist for reviewing NumPy code  
Where is each array created? Look at np.array, *_like, .copy(), file loaders and slicing. Ask what dtype comes out, especially when it’s inherited from the input.  
Where do values get written back? Item assignment, += and out= all keep the destination’s dtype.  
Are there small or unsigned ints? Check for subtraction, large products and long sums.  
Are masks actually bool? Look for masks built from 0/1 data or loaded from disk.  
Do astype(int) calls mean truncation or rounding?  
At function boundaries, use explicit dtype= and add a cheap check like assert w.dtype.kind == 'f'. Then add a test that feeds integer inputs on purpose, like the one you wrote.
