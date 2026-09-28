package ai

import (
	"neural-network/dataset"
)

type StandardizedData struct {
	data []float64
	cols int
}

func NewStandardizedData(ds *dataset.DataSet) *StandardizedData {
	p := NewStandardizationParams(ds)
	src := ds.Features()
	cols := ds.Cols()

	out := make([]float64, len(src))
	for i, v := range src {
		out[i] = p.Standardize(v, i % cols)
	}

	return &StandardizedData{data: out, cols: cols}
}

