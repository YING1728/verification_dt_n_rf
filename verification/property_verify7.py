from to_solver import solver
import pickle # using to save and load object


def main():

    # Test on: N1_9
    #Constraints: 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ 3.141592, −3.141592 ≤ ψ ≤ 3.141592, 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200.
    # Desired output property: the scores for “strong right” and “strong left” are never the minimal scores.

    p7 = ["f0 >= 0", "f0 <= 60760",
          "f1 >= -3.141592", "f1 <= 3.141592",
          "f2 >= -3.141592", "f2 <= 3.141592",
          "f3 >= 100", "f3 <= 1200",
          "f4 >= 0", "f4 <= 1200", "Not(Or(lmin == 4,lmin ==3))"]

    with open("../paths_generation/paths_dt/tree_paths_19.txt", "rb") as fp:
        paths = pickle.load(fp)

    with open("../paths_generation/features.txt", "rb") as fp:
        f = pickle.load(fp)
    solver(paths, f, p7)



if __name__ == "__main__":
    main()
