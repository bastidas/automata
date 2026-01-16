import pylinkage as pl



def make_demo_linkage():
    ############
    # Create a four-bar linkage
    ############
    # Main motor
    crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1)
    # Close the loop
    pin = pl.Revolute(
        3, 2, joint0=crank, joint1=(3, 0), 
        distance0=3, distance1=1
    )

    my_linkage = pl.Linkage(joints=(crank, pin))

    locus = my_linkage.step()

    crank.name = "B"
    
    """
    Here we are actually doing the following:

    0, 1: x and y initial coordinates of the tail of the crank link.
    joint0: the position of the parent Joint to link with, here it is a fixed point in 
    space. The pin will be created on the position of the parent, which is the head of
    the crank link.
    angle: the crank will rotate with this angle, in radians, at each iteration.
    distance: distance to keep constant between crank link tail and head.
    Now we add a pin joint to close the kinematic loop.
    """
    pin.name = "C"

    """
    In human language, here is what is happening:

    joint0, joint1: first and second Joints you want to link to, the order is not important.
    distance0, distance1: distance to keep constant between this joint and his two parents.
    And here comes the trick: Why do we specify initial coordinates 3, 2? They even seem
    incompatible with distance to parents/parents' positions!

    This explanation is simple: mathematically a pin joint the intersection of two circles.
    The intersection is often two points. To choose the starting point, we calculate both 
    intersection (when possible), then we keep the intersection closer to the previous
   position as the solution.
    """
    # Linkage can also have names
    my_linkage.name = "Four-bar linkage"

    # pl.show_linkage(my_linkage)
    return my_linkage


def optimization(my_linkage):
    """
    
    Now, we want automatic optimization of our linkage, using a certain criterion.
      Let's find a four-bar linkage that makes a quarter of a circle. It is a 
      common problem if you want to build a windshield wiper for instance.

    Our objective function, often called the fitness function, is the following:
    """
    # Save initial state for later reset
    init_pos = my_linkage.get_coords()
    init_constraints = my_linkage.get_num_constraints()

    @pl.kinematic_minimization
    def fitness_func(loci, **_kwargs):
        """
        Return how fit the locus is to describe a quarter of circle.

        It is a minimization problem and the theoretic best score is 0.
        """
        # Locus of the Joint 'pin', last in linkage order
        tip_locus = tuple(x[-1] for x in loci)
        # We get the bounding box
        curr_bb = pl.bounding_box(tip_locus)
        # We set the reference bounding box, in order (min_y, max_x, max_y, min_x)
        ref_bb = (0, 5, 3, 0)
        # Our score is the square sum of the edge distances
        return sum((pos - ref_pos) ** 2 for pos, ref_pos in zip(curr_bb, ref_bb))
    """
    Please note that it is a minimization problem, with 0 as lower bound. On the
      first line, you notice a decorator; which plays a crucial role:

    The decorator arguments are (linkage, constraints), it can also receive init_pos
    It sets the linkage with the constraints.
    Then it verifies if the linkage can do a complete crank turn.
    If it can, pass the arguments and the resulting loci (path of joints) to the 
    decorated function.
    If not, return the penalty. In a minimization problem the penalty will be 
    float('inf').
    The decorated function should return the score of this linkage.
    With this constraint, the best theoretic score is 0.0.

    Let's start with a candide optimization, the trial-and-error method.

    Here it is a serial test of switches.
    """
    # Exhaustive optimization as an example ONLY
    score, position, coord = pl.trials_and_errors_optimization(
        eval_func=fitness_func,
        linkage=my_linkage,
        divisions=25,
        n_results=1,
        order_relation=min,
    )[0]

    print(f"Best score: {score}")
    print(f"Best position: {position}")
    print(f"Best coordinates: {coord}")
    # # Reset linkage to best found position
    # #my_linkage.set_coords(coord)
    # # Show the optimized linkage
    # pl.show_linkage(my_linkage)
    # print("Optimization done.")



    """
     Here the problem is simple enough, so that method takes only a few s
     econds and returns 0.05.

    However, with more complex linkages, you need something more robust and more
    efficient. Then we will use particle swarm optimization.

    """
    #Reset the linkage to initial state before optimization
    my_linkage.set_num_constraints(init_constraints)
    my_linkage.set_coords(init_pos)

    # Generate bounds for the optimization search space
    bounds = pl.generate_bounds(my_linkage.get_num_constraints())

    score, position, coord = pl.particle_swarm_optimization(
        eval_func=fitness_func,
        linkage=my_linkage,
        bounds=bounds,
        order_relation=min,
    )[0]
    print(f"Best score: {score}")
    print(f"Best position: {position}")

    print(f"Best coordinates: {coord}")
    return score


if __name__ == "__main__":
    demo_linkage = make_demo_linkage()
    optimization(demo_linkage)
    print("Demo complete.")
