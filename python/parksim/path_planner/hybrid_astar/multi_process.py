import multiprocessing
import time

NUMBER_OF_PROCESSES = 5
NUMBER_OF_ITERATIONS = 5
N = 100000000  # 100 million


def sum_all_numbers(n):
    """
    Sums all the numbers from zero to n.

    :param n: The upper bound of numbers to be summed
    :return: The sum of all the numbers from 0 to n
    """

    total_sum = sum(range(n + 1))
    return print("Sum: " + str(total_sum))


def without_multiprocessing():
    print("Starting function without multiprocessing.")
    for i in range(NUMBER_OF_ITERATIONS):
        sum_all_numbers(N)


def with_multiprocessing():
    print("Starting function with multiprocessing.")
    jobs = []

    for i in range(NUMBER_OF_PROCESSES):
        process = multiprocessing.Process(
            target=sum_all_numbers,
            args=(N,)
        )
        jobs.append(process)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()


def main():
    print("Summing all numbers between 0 and " + str(N) + ".\n")

    start_time = time.time()
    without_multiprocessing()
    print("--- Function without multiprocessing took %s seconds ---\n" % (
            time.time() - start_time))

    start_time = time.time()
    with_multiprocessing()
    print("--- Function with multiprocessing took %s seconds ---" % (
            time.time() - start_time))


if __name__ == "__main__":
    main()