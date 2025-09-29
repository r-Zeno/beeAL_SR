#include <iostream>
#include "cnpy.h"

int main()
{

    cnpy::NpyArray array = cnpy::npy_load("neuron_rates.npy")
    double* data = array.data<double>()

    

}

