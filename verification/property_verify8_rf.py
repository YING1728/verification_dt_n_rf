from to_solver import solver
import pickle # using to save and load object


def main():

    # Test on: N2_9
    # 0 ≤ ρ ≤ 60760, -3.141592 ≤ θ ≤ -0.75·3.141592, -0.1 ≤ ψ ≤ 0.1, 600 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200
    # Desired output property: the score for “weak left” is minimal or the score for COC is minimal.

    p8 = ["f0 >= 0", "f0 <= 60760",
          "f1 >= -3.141592", "f1 <= -0.75 * 3.141592",
          "f2 >= -0.1", "f2 <= 0.1",
          "f3 >= 600", "f3 <= 1200",
          "f4 >= 600", "f4 <= 1200", "Or(lmin == 1,lmin ==0)"]



    with open("../paths_generation/paths_rf/combined_paths_2_9.txt", "rb") as fp:
        paths = pickle.load(fp)

    with open("../paths_generation/features.txt", "rb") as fp:
        f = pickle.load(fp)

    solver(paths, f, p8)


if __name__ == "__main__":
    main()
