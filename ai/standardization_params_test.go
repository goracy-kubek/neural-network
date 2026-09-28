package ai

import (
	"neural-network/dataset"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStandardizationData(t *testing.T) {
	features := []float64{
		5.1, 3.5, 1.4, 0.2,
		4.9, 3.0, 1.4, 0.2,
		7.0, 3.2, 4.7, 1.4,
		6.4, 3.2, 4.5, 1.5,
		6.3, 3.3, 6.0, 2.5,
	}
	targets := []string{"setosa", "setosa", "versicolor", "versicolor", "virginica"}

	res := NewStandardizationParams(dataset.NewDataSetValues(features, 5, targets))

	assert.InDeltaSlice(t, []float64{5.94, 3.24, 3.6, 1.16}, res.mean, 1e-9)
	assert.InDeltaSlice(t, []float64{0.806, 0.162, 1.869, 0.873}, res.stdDev, 1e-3)
}
