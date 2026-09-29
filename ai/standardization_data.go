package ai

import (
	"iter"
	"neural-network/dataset"
	"slices"
)

type StandardizedDataLine struct {
	Data []float64
}

type StandardizedData struct {
	data []float64
	cols int
}

func (sd *StandardizedData) DataLine(i int) *StandardizedDataLine {
	return &StandardizedDataLine{
		Data: slices.Clone(sd.data[i * sd.cols : (i + 1) * sd.cols]),
	}
}

func NewStandardizedData(ds *dataset.DataSet, p *StandardizationParams) *StandardizedData {
	src := ds.Features()
	cols := ds.Cols()

	out := make([]float64, len(src))
	for i, v := range src {
		out[i] = p.Standardize(v, i % cols)
	}

	return &StandardizedData{data: out, cols: cols}
}

func (sd *StandardizedData) Rows() int {
	if sd.cols == 0 {
		return 0
	}
	return len(sd.data) / sd.cols
}

func (sd *StandardizedData) Lines() iter.Seq2[int, *StandardizedDataLine] {
	return func(yield func(int, *StandardizedDataLine) bool) {
		for i := range sd.Rows() {
			if !yield(i, sd.DataLine(i)) {
				return
			}
		}
	}
}