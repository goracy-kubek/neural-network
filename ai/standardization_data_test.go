package ai

import (
	"neural-network/dataset"
	"testing"

	"github.com/stretchr/testify/assert"
)


func TestStandardization(t *testing.T) {
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

	res := NewStandardizedData(data)

	expected := []float64{
		-1.04, 1.60, -1.18, -1.10,
		-1.29, -1.48, -1.18, -1.10,
		1.31, -0.25, 0.59, 0.27,
		0.57, -0.25, 0.48, 0.39,
		0.45, 0.37, 1.28, 1.53,
	}

	assert.Equal(t, 4, res.cols)
	assert.InDeltaSlice(t, expected, res.data, 1e-2)
}