from to_solver import solver
import pickle # using to save and load object


def main():

    # Test on: N3_3
    # 2000 ≤ ρ ≤ 7000, −0.4 ≤ θ ≤ −0.14, −3.141592 ≤ ψ ≤ −3.141592 + 0.01, 100 ≤ vown ≤ 150, 0 ≤ vint ≤ 150.
    #  Desired output property: the score for “strong left” is minimal.

    p9 = ["f0 >= 2000", "f0 <= 7000",
          "f1 >= -0.4", "f1 <= -0.14",
          "f2 >= -3.141592", "f2 <= -3.141592 + 0.01",
          "f3 >= 100", "f3 <= 150",
          "f4 >= 0", "f4 <= 150", "lmin == 3"]

    with open("../paths_generation/paths_dt/tree_paths_33.txt", "rb") as fp:
        paths = pickle.load(fp)

    with open("../paths_generation/features.txt", "rb") as fp:
        f = pickle.load(fp)

    solver(paths, f, p9)


if __name__ == "__main__":
    main()
