
# Test for optimizing a four-bar linkage to match a rectangular path
import pylinkage as pl
import numpy as np

def create_test_linkage():
	"""
	Create a base linkage: crank at (0,0), length 1, with attached link as in ref.txt and demo.py.
	Crank is at origin (0,0), output pin is attached as in the reference example.
	"""
	crank = pl.Crank(
		x=0, y=0,  # Crank at origin
		joint0=(0, 0),
		angle=0.31,
		distance=1,
		name="Crank"
	)
	output = pl.Revolute(
		x=3, y=2,
		joint0=crank,
		joint1=(3, 0),
		distance0=3,
		distance1=1,
		name="Output"
	)
	
	linkage = pl.Linkage(
		joints=(crank, output),
		order=(crank, output),
	)

	#pl.show_linkage(linkage)
	return linkage 

# Fitness function as in ref.txt, but ensure all values are floats (no None)
def rectangle_fitness_base(loci, target=(0.0, 5.0, 2.0, 3.0)):
	"""
	Fitness: minimize distance from a target rectangle bounding box.
	Only uses 2-tuples of floats, skips any with None or wrong length.
	Args:
		loci: List of joint positions at each step
		target: Target bounding box as (min_y, max_x, max_y, min_x)
	"""
	output_path = [step[-1] for step in loci if isinstance(step[-1], tuple) and len(step[-1]) == 2 and all(isinstance(x, (float, int)) and x is not None for x in step[-1])]
	if not output_path:
		return float('inf')
	bbox = pl.bounding_box(output_path)
	return sum((actual - target_val) ** 2 for actual, target_val in zip(bbox, target))

def test_optimize_rectangle_path(target=(0.0, 5.0, 2.0, 3.0)):
	linkage = create_test_linkage()
	# Get constraints as a flat list of floats (skip tuples/None)
	raw_constraints = linkage.get_num_constraints()
	constraints = []
	for x in raw_constraints:
		if isinstance(x, (float, int)):
			constraints.append(float(x))
		elif isinstance(x, tuple):
			constraints.extend(float(y) for y in x if y is not None)
	# Generate bounds as lists of floats, ensure all elements are floats
	bounds_raw = pl.generate_bounds(constraints)
	def to_float_list(arr):
		if hasattr(arr, 'tolist'):
			arr = arr.tolist()
		return [float(x) for x in arr]
	bounds = (to_float_list(bounds_raw[0]), to_float_list(bounds_raw[1]))
	
	# Track optimization progress
	history = []
	
	# Create a closure that captures target
	@pl.kinematic_minimization
	def rectangle_fitness(loci, **kwargs):
		return rectangle_fitness_base(loci, target=target)
	
	def tracking_fitness(*args, **kwargs):
		score = rectangle_fitness(*args, **kwargs)
		constraints = args[1] if len(args) > 1 else None
		history.append((score, constraints))
		return score
	
	results = pl.particle_swarm_optimization(
		eval_func=tracking_fitness,
		linkage=linkage,
		bounds=bounds,
		n_particles=50,
		iters=100,
		order_relation=min,
	)
	best_score, best_constraints, best_coords = results[0]
	# Apply best constraints
	linkage.set_num_constraints(best_constraints)
	# Check the bounding box
	loci = linkage.step()
	output_path = [
		(float(pt[0]), float(pt[1]))
		for step in loci
		for pt in [step[-1]]
		if (
			isinstance(pt, tuple)
			and len(pt) == 2
			and pt[0] is not None and pt[1] is not None
			and all(isinstance(x, (float, int)) for x in pt)
		)
	]
	bbox = pl.bounding_box(output_path)
	
	# Assert the score is reasonably low (tolerance may be adjusted)
	assert best_score < 1.0, f"Optimization did not converge: score={best_score} bbox={bbox}"
	
	# Visualize optimization progress
	scores = [h[0] for h in history]
	print(f"Score improvement: {scores[0] if scores else 'N/A'} -> {scores[-1] if scores else 'N/A'}")
	try:
		import matplotlib.pyplot as plt
		if scores:
			plt.plot(scores)
			plt.title("Optimization Progress (Score)")
			plt.xlabel("Iteration")
			plt.ylabel("Score")
			plt.show()
	except ImportError:
		print("matplotlib not available for plotting.")
	
	# Show final optimized linkage with bounding box overlay
	print(f"Test best score: {best_score}\nBounding box: {bbox}")
	print(f"Target bounding box: {target}")
	
	# Visualize linkage with bounding box
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	
	# Plot the linkage path
	loci_all = linkage.step()
	output_points = [(step[-1][0], step[-1][1]) for step in loci_all 
	                 if step[-1][0] is not None and step[-1][1] is not None]
	if output_points:
		output_x = [pt[0] for pt in output_points]
		output_y = [pt[1] for pt in output_points]
		ax.plot(output_x, output_y, 'b-', linewidth=2, label='Output path')
	
	# Draw actual bounding box
	min_y, max_x, max_y, min_x = bbox
	ax.plot([min_x, max_x, max_x, min_x, min_x], 
	        [min_y, min_y, max_y, max_y, min_y], 
	        'r--', linewidth=2, label=f'Actual bbox')
	
	# Draw target bounding box
	min_y_t, max_x_t, max_y_t, min_x_t = target
	ax.plot([min_x_t, max_x_t, max_x_t, min_x_t, min_x_t], 
	        [min_y_t, min_y_t, max_y_t, max_y_t, min_y_t], 
	        'g:', linewidth=2, label=f'Target bbox')
	
	ax.set_aspect('equal')
	ax.legend()
	ax.grid(True, alpha=0.3)
	ax.set_title('Optimized Linkage Path with Bounding Boxes')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	plt.show()
	
	# Show the standard pylinkage visualization (may fail if linkage is at edge of buildable)
	try:
		pl.show_linkage(linkage)
	except Exception as e:
		print(f"Could not show animated linkage: {e}")


if __name__ == "__main__":
	target = (0.0, 5.0, 2.0, 4.5)  # min_y, max_x, max_y, min_x
	test_optimize_rectangle_path(target=target)
	print("Optimization test complete.")