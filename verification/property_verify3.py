from to_solver import solver
import pickle # using to save and load object


def main():

    # Tested on: all networks except N1_7, N1_8, and N1_9.
    # Constraints: 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, ψ ≥ 3.10, vown ≥ 980, vint ≥ 960.
    # The score for COC is not the minimal score
    p3 = ["f0 >= 1500", "f0 <= 1800",
          "f1 >= -0.06", "f1 <= 0.06",
          "f2 >= 3.1",
          "f3 >= 980",
          "f4 >= 960",  "Not(lmin == 0)"]
    # enter the correponding path file: "../paths_generation/tree_paths_xy.txt", "rb"
    # all networks except N1_7, N1_8, and N1_9.
    with open("../paths_generation/paths_dt/tree_paths_59.txt", "rb") as fp:
        paths = pickle.load(fp)

    with open("features.txt", "rb") as fp:
        f = pickle.load(fp)

    solver(paths, f, p3)


if __name__ == "__main__":
    main()
