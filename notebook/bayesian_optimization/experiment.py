import json

FILE_PATH = "./parameters.json"


def experiment(x1, x2):
    return -(x1**2) - (x2 - 1) ** 2 + 1


def main():
    with open(FILE_PATH, "r") as f:
        parameters = json.load(f)
    result = experiment(**parameters)
    print(result)


if __name__ == "__main__":
    main()
