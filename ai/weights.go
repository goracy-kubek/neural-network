package ai

import (
	"math"
	"math/rand/v2"
	"neural-network/dataset"
)

type WeightDataLine struct {
	Data []float64
	Bias float64
	Target string
}

type WeightData struct {
	data []float64
	cols    int
	biases  []float64
	targets []string
}

func (w *WeightData) Line(i int) *WeightDataLine {
	return &WeightDataLine {
		w.data[i * w.cols : (i + 1) * w.cols],
		w.biases[i],
		w.targets[i],
	}
}

func NewWeights(ds *dataset.DataSet) *WeightData {
	unqSize := len(ds.UniqueTargets())

	rng := randomRange(ds.Cols(), unqSize)
	size := unqSize * ds.Cols()

	weights := make([]float64, size)
	for i := range size {
		weights[i] = (rand.Float64()*2 - 1) * rng
	}

	return &WeightData{
		data: weights,
		cols:    ds.Cols(),
		biases:  make([]float64, unqSize),
		targets: ds.UniqueTargets(),
	}
}

func randomRange(input int, output int) float64 {
	return math.Sqrt(6 / float64(input+output))
}
