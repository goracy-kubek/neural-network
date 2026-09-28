package ai

import "neural-network/dataset"

type Targets struct {
	targets []string
}

func NewOneHot(ds *dataset.DataSet) *Targets {
	return &Targets{
		targets: ds.UniqueTargets(),
	}
}