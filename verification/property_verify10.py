from to_solver import solver
import pickle # using to save and load object


def main():

    # Test on: N4_5
    # 36000 ≤ ρ ≤ 60760, 0.7 ≤ θ ≤ 3.141592, -3.141592 ≤ ψ ≤ -3.141592 + 0.01, 900 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200.
    # Desired output: the score for COC is minimal.

    p10 = ["f0 >= 36000", "f0 <= 60760",
           "f1 >= 0.7", "f1 <= 3.141592",
           "f2 >= -3.141592", "f2 <= -3.141592 + 0.01",
           "f3 >= 900", "f3 <= 1200",
           "f4 >= 600", "f4 <= 1200", "lmin == 0"]

    with open("../paths_generation/paths_dt/tree_paths_45.txt", "rb") as fp:
        paths = pickle.load(fp)

    with open("../paths_generation/features.txt", "rb") as fp:
        f = pickle.load(fp)

    solver(paths, f, p10)

if __name__ == "__main__":
    main()
