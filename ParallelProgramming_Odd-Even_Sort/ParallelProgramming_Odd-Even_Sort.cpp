#include <iostream>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <string>

//cd C:\Users\deyko\source\repos\ParallelProgramming_Odd-Even_Sort\x64\Release
//mpiexec -n 2 ParallelProgramming_Odd-Even_Sort.exe 100000

std::vector<int> generateRandomArray(int size) {
	std::vector<int> array(size);
	for (int i = 0; i < size; ++i) {
		array[i] = std::rand() % 1000;
	}
	return array;
}

void printProcessInfo(int rank, int size, int arraySize) {
	char processorName[MPI_MAX_PROCESSOR_NAME];
	int nameLen;
	MPI_Get_processor_name(processorName, &nameLen);

	if (rank == 0) {
		std::cout << "Size: " << arraySize << std::endl;
	}
	std::cout << "Process " << rank << " on " << processorName << std::endl;
}

void distributeData(const std::vector<int>& globalArray, std::vector<int>& localArray,
	int elementsPerProcess, int rootRank) {
	MPI_Scatter(globalArray.data(), elementsPerProcess, MPI_INT,
		localArray.data(), elementsPerProcess, MPI_INT, rootRank, MPI_COMM_WORLD);
}

void sortLocalArray(std::vector<int>& array) {
	std::sort(array.begin(), array.end());
}

void exchangeAndMerge(std::vector<int>& localArray, const std::vector<int>& receivedArray,
	std::vector<int>& mergedArray, bool keepSmaller) {
	std::merge(localArray.begin(), localArray.end(),
		receivedArray.begin(), receivedArray.end(), mergedArray.begin());
	if (keepSmaller) {
		std::copy(mergedArray.begin(), mergedArray.begin() + localArray.size(), localArray.begin());
	}
	else {
		std::copy(mergedArray.begin() + localArray.size(), mergedArray.end(), localArray.begin());
	}
}

void performOddEvenPhase(std::vector<int>& localArray, int elementsPerProcess, int rank, int size, int phase) {
	int partner;
	std::vector<int> receivedArray(elementsPerProcess);
	std::vector<int> mergedArray(2 * elementsPerProcess);
	bool evenPhase = (phase % 2 == 0);

	if ((evenPhase && rank % 2 == 0) || (!evenPhase && rank % 2 == 1)) {
		partner = rank + 1;
		if (partner < size) {
			MPI_Sendrecv(localArray.data(), elementsPerProcess, MPI_INT, partner, 0,
				receivedArray.data(), elementsPerProcess, MPI_INT, partner, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			exchangeAndMerge(localArray, receivedArray, mergedArray, true);
		}
	}
	else {
		partner = rank - 1;
		if (partner >= 0) {
			MPI_Sendrecv(localArray.data(), elementsPerProcess, MPI_INT, partner, 0,
				receivedArray.data(), elementsPerProcess, MPI_INT, partner, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			exchangeAndMerge(localArray, receivedArray, mergedArray, false);
		}
	}
}

void performOddEvenSort(std::vector<int>& localArray, int elementsPerProcess, int rank, int size) {
	for (int phase = 0; phase < size; ++phase) {
		performOddEvenPhase(localArray, elementsPerProcess, rank, size, phase);
	}
}

void gatherResults(const std::vector<int>& localArray, std::vector<int>& sortedArray,
	int elementsPerProcess, int rootRank) {
	MPI_Gather(localArray.data(), elementsPerProcess, MPI_INT,
		sortedArray.data(), elementsPerProcess, MPI_INT, rootRank, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int arraySize = 100;
	if (argc > 1) {
		arraySize = std::atoi(argv[1]);
	}

	printProcessInfo(rank, size, arraySize);

	double start_time = MPI_Wtime();
	int elementsPerProcess = arraySize / size;
	std::vector<int> globalArray;

	if (rank == 0) {
		std::srand(static_cast<unsigned int>(std::time(nullptr)));
		globalArray = generateRandomArray(arraySize);
	}

	std::vector<int> localArray(elementsPerProcess);
	distributeData(globalArray, localArray, elementsPerProcess, 0);

	sortLocalArray(localArray);

	performOddEvenSort(localArray, elementsPerProcess, rank, size);

	std::vector<int> sortedArray;
	if (rank == 0) {
		sortedArray.resize(arraySize);
	}
	gatherResults(localArray, sortedArray, elementsPerProcess, 0);

	double end_time = MPI_Wtime();

	if (rank == 0) {
		std::cout << "Time: " << end_time - start_time << std::endl;
	}

	MPI_Finalize();
	return 0;
}