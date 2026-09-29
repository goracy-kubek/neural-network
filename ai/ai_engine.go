package ai

import (
	"math"
	"neural-network/dataset"
	"slices"
)

func Run(ds *dataset.DataSet) []string {
	m := NewModelData(ds)
	sd := NewStandardizedData(ds, m.Standardized)

	res := make([]string, ds.Rows())

	for i, line := range sd.Lines() {
		res[i] = logitResult(line, m)
	}

	return res
}

func logitResult(sd *StandardizedDataLine, m *ModelData) string {
	targets := m.WeightTargets()

	logits := make([]float64, len(targets))
	for i, weightLine := range m.WeightLines() {
		logits[i] = summarizeLogitData(sd, weightLine)
	}

	return m.Target(bestProbabilityIdx(logits))
}

func summarizeLogitData(sd *StandardizedDataLine, wl *WeightDataLine) float64 {
	var sum float64

	for i, data := range sd.Data {
		sum += (data * wl.Data[i])
	}

	return sum + wl.Bias
}

func softmax(logits []float64) []float64 {
	ln := len(logits)
	max := slices.Max(logits)
	exps := make([]float64, ln)
	var sumExp float64

	for i, logit := range logits {
		exp := math.Exp(max - logit)
		sumExp += exp
		exps[i] = exp
	}

	res := make([]float64, ln)
	for i, ex := range exps {
		res[i] = ex / sumExp
	}

	return res
}

func bestProbabilityIdx(logits []float64) int {
	if len(logits) == 0 {
        return -1
    }

    idx := 0
    for i, v := range logits {
        if v > logits[idx] {
            idx = i
        }
    }

    return idx
}