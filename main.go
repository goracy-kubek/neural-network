package main

import "neural-network/dataset"

func main() {
	dset, err := dataset.NewDataSetFile("dataset/data/IrisTest.csv", -1, true)
	if err != nil {
		println(err.Error())
	}

	println(dset)
}
