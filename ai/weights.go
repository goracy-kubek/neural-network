package ai

import (
	"math"
	"math/rand/v2"
	"neural-network/dataset"
)

type Weights struct {
	weights []float64
	cols int
	biases []float64
}

func NewWeights(ds *dataset.DataSet) *Weights {
	unqSize := len(ds.UniqueTargets())

	rng := randomRange(ds.Cols(), unqSize)
	size := unqSize * ds.Cols()
	
	weights := make([]float64, size)
	for i := range size {
		weights[i] = (rand.Float64() * 2 - 1) * rng
	}

	return &Weights{
		weights: weights,
		cols: ds.Cols(),
		biases: make([]float64, unqSize),
	}
}

func randomRange(input int, output int) float64 {
	return math.Sqrt(6 / float64(input + output))
}