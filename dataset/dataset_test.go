package dataset

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var ds, err = NewDataSetFile("testdata/IrisTest.csv", -1, true)

func TestParsing(t *testing.T) {
	require.NoError(t, err)
	require.NotNil(t, ds)

	assert.Equal(t, 5, ds.Rows())
	assert.Equal(t, 4, ds.Cols())
	assert.Equal(t, []float64{7.0,3.2,4.7,1.4}, ds.Row(2))
	assert.Equal(t, []float64{6.3,3.3,6.0,2.5}, ds.Row(4))
	assert.Equal(t, "versicolor", ds.Target(2))
	assert.Equal(t, "virginica",  ds.Target(4))
}

func TestColumnFetching(t *testing.T) {
	assert.Equal(t, []float64{5.1, 4.9, 7.0, 6.4, 6.3}, ds.Col(0))
	assert.Equal(t, []float64{3.5, 3.0, 3.2, 3.2, 3.3}, ds.Col(1))
	assert.Equal(t, []float64{1.4, 1.4, 4.7, 4.5, 6.0}, ds.Col(2))
	assert.Equal(t, []float64{0.2, 0.2, 1.4, 1.5, 2.5}, ds.Col(3))
}