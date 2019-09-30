
from to_solver import solver
import pickle # using to save and load object


def main():

    #Test on: N1_1
    #Constraints: 12000 ≤ ρ ≤ 62000, (0.7 ≤ θ ≤ 3.141592)∨(−3.141592≤ θ ≤ -0.7),
    # -3.141592 ≤ ψ ≤ -3.141592 + 0.005, 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200.
    #  the score of ``Clear-of-Conflict" is the minimal )\vee (-3.141592 $\leq\theta\leq$ -0.7), -3.141592 $\leq \psi \leq $ -3.141592 + 0.005, 100 $\leq v_{own} \leq$ 1200, 0 $\leq v_{int} \leq$ 1200.
    # the property is divided into two sub-properties that can be check seperately.
    p6_a = ["f0 >= 12000", "f0 <= 62000",
          "f1 >= 0.7", "f1 <= 3.141592",
          "f2 >= -3.141592", "f2 <= -3.141592 + 0.005",
          "f3 >= 100", "f3 <= 1200",
          "f4 >= 0", "f4 <= 1200", "lmin == 0"]

    p6_b = ["f0 >= 12000", "f0 <= 62000",
            "f1 >= -3.141592", "f1 <= -0.7",
            "f2 >= -3.141592", "f2 <= -3.141592 + 0.005",
            "f3 >= 100", "f3 <= 1200",
            "f4 >= 0", "f4 <= 1200", "lmin == 0"]

    with open("../paths_generation/paths_dt/tree_paths_11.txt", "rb") as fp:
        paths = pickle.load(fp)

    with open("../paths_generation/features.txt", "rb") as fp:
        f = pickle.load(fp)

    #solver(paths, f, p6_a)
    solver(paths, f, p6_b)



if __name__ == "__main__":
    main()
