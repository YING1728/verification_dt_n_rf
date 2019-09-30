from to_solver import solver
import pickle # using to save and load object


def main():

    #Test on: N1_1
    #constraints: 250 ≤ ρ ≤ 400, 0.2 ≤ θ ≤ 0.4, -3.141592 ≤ ψ ≤ -3.141592 + 0.005, 100 ≤ vown ≤ 400, 0 ≤ vint ≤ 400
    #the score of “strong right” is the minimal

    p5 = ["f0 >= 250", "f0 <= 400",
          "f1 >= 0.2", "f1 <= 0.4",
          "f2 >= -3.141592", "f2 <= -3.141592+0.005",
          "f3 >= 100", "f3 <= 400",
          "f4 >= 0", "f4 <= 400", "lmin == 4"]

    with open("../paths_generation/paths_dt/tree_paths_11.txt", "rb") as fp:
        paths = pickle.load(fp)

    with open("features.txt", "rb") as fp:
        f = pickle.load(fp)

    solver(paths, f, p5)

if __name__ == "__main__":
    main()
