package main

import (
	"fmt"
	"neural-network/ai"
	"neural-network/dataset"
)

func main() {
	ds, err := dataset.NewDataSetFile("dataset/testdata/IrisTest.csv", -1, true)
	if err != nil {
		println(err.Error())
	}

	res := ai.Run(ds)
	checkResult(ds, res)
}

func checkResult(ds *dataset.DataSet, res []string) {
	for i, tg := range ds.Targets() {
		fmt.Printf("%d: expected %s, got %s\n", i, tg, res[i])
	}
}