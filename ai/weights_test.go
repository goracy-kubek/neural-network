package ai

import (
	"math"
	"neural-network/dataset"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewWeights(t *testing.T) {
	data := dataset.NewDataSetValues(
		[]float64{
			5.1, 3.5, 1.4, 0.2,
			4.9, 3.0, 1.4, 0.2,
			7.0, 3.2, 4.7, 1.4,
			6.4, 3.2, 4.5, 1.5,
			6.3, 3.3, 6.0, 2.5,
		},
		5,
		[]string{
			"setosa",
			"setosa",
			"versicolor",
			"versicolor",
			"virginica",
		},
	)
	
	w := NewWeights(data)
	limit := math.Sqrt(6.0 / 7.0)

	assert.Len(t, w.weights, 12)
	assert.Len(t, w.biases, 3)
	for _, v := range w.weights {
		assert.LessOrEqual(t, math.Abs(v), limit)
	}
}