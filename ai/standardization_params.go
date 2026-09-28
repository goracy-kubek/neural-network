package ai

import (
	"math"
	"neural-network/dataset"
)

type StandardizationParams struct {
	mean   []float64
	stdDev []float64
}

func (p *StandardizationParams) Standardize(val float64, col int) float64 {
	if p.stdDev[col] == 0 {
		return 0
	}
	
	return (val - p.mean[col]) / p.stdDev[col]
}

func NewStandardizationParams(ds *dataset.DataSet) *StandardizationParams {
	src := ds.Features()
	rows, cols := ds.Rows(), ds.Cols()

	mean := make([]float64, cols)
	for i, v := range src {
		mean[i % cols] += v
	}
	for c := range mean {
		mean[c] /= float64(rows)
	}

	stdDev := make([]float64, cols)
	for i, v := range src {
		d := v - mean[i % cols]
		stdDev[i % cols] += d * d
	}
	for c := range stdDev {
		stdDev[c] = math.Sqrt(stdDev[c] / float64(rows))
	}

	return &StandardizationParams{mean: mean, stdDev: stdDev}
}
