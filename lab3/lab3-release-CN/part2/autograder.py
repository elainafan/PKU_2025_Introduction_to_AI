def grade_func(step):
    full_ = 50
    pass_ = 90
    if step > pass_:
        return 0
    if step < full_:
        return 100
    return (pass_ - full_) / 40 * (pass_ - step) + 60


if __name__ == "__main__":
    file_name = "res.txt"
    print("Autograder is running...")
    with open("res.txt", "r", encoding="utf-8") as file:
        step = file.read()
        print(f"Score: {grade_func(int(step))}")