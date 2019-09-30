from to_solver import solver
import pickle # using to save and load object


def main():

    # Tested on: Nx_y for all x ≥ 2 and for all y.
    # Constraints: ρ ≥ 55947.691, vown ≥ 1145, vint ≤ 60.
    # The score for COC is not the maximal score : Not(lmax == 0)

    p2 = ["f0 >= 55947.691",
          "f3 >= 1145",
          "f4 <= 60", "Not(lmax == 0)"]

    # enter the correponding path file: "../paths_generation/tree_paths_xy.txt", "rb"
    # Nx,y for all x ≥ 2 and for all y.
    with open("../paths_generation/paths_dt/tree_paths_59.txt", "rb") as fp:
        paths = pickle.load(fp)
    with open("../paths_generation/features.txt", "rb") as fp:
        f = pickle.load(fp)

    solver(paths, f, p2)

if __name__ == "__main__":
    main()
