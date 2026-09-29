package ai

import (
	"iter"
	"neural-network/dataset"
)

// {
//   "standardized": {..   ЭыефтвфквшяувЭЖ Х

//     "mean":    [5.94, 3.24, 3.60, 1.16],
//     "std_dev": [0.806, 0.162, 1.869, 0.873]
//   },
//   "weights": {
//     "data": [
//       -1.45, -0.01, -1.71, -1.16,
//        2.41, -1.15,  0.30, -0.78,
//       -1.06,  1.35,  1.42,  2.44
//     ],
//     "cols":    4,
//     "biases":  [-0.04, 1.15, -1.10],
//     "targets": ["setosa", "versicolor", "virginica"]
//   }
// }

type ModelData struct {
	Standardized *StandardizationParams
	Weights      *WeightData
}

func NewModelData(ds *dataset.DataSet) *ModelData {
	return &ModelData{
		Standardized: NewStandardizationParams(ds),
		Weights:      NewWeights(ds),
	}
}

func (m *ModelData) WeightTargets() []string {
	return m.Weights.targets
}

func (m *ModelData) Target(i int) string {
	return m.Weights.targets[i]
}

func (m *ModelData) WeightLines() iter.Seq2[int, *WeightDataLine] {
	return func(yield func(int, *WeightDataLine) bool) {
		for i := range len(m.Weights.targets) {
			if !yield(i, m.Weights.Line(i)) {
				return
			}
		}
	}
}
