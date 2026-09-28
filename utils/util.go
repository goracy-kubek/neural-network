package utils

func Sum(arr []float64) float64 {
	var sum float64

	for _, e := range arr {
		sum += e
	}

	return sum
}

func Average(arr []float64) float64 {
	return Sum(arr) / float64(len(arr))
}