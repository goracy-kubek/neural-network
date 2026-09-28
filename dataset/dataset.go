package dataset

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"strconv"
)

type DataSet struct {
	featuresRow []float64
	rows        int
	targets     []string
	uniqueTargets []string
}

func (d *DataSet) Features() []float64 { return d.featuresRow }
func (d *DataSet) Rows() int           { return d.rows }
func (d *DataSet) Target(i int) string { return d.targets[i] }
func (d *DataSet) Targets() []string { return d.targets }
func (d *DataSet) UniqueTargets() []string { return d.uniqueTargets }
func (d *DataSet) Row(i int) []float64 {
	cols := d.Cols()

	return d.featuresRow[i*cols : (i+1)*cols]
}
func (d *DataSet) Cols() int {
	if d.rows == 0 {
		return 0
	}

	return len(d.featuresRow) / d.rows
}
func (d *DataSet) Col(i int) []float64 {
	var res []float64
	for idx, e := range d.featuresRow {
		if(idx % d.Cols() == i) {
			res = append(res, e)
		} 
	}

	return res
}

func NewDataSetValues(featuresRow []float64, rows int, targets []string) *DataSet {
	return &DataSet{
		featuresRow: featuresRow,
		rows:        rows,
		targets:     targets,
		uniqueTargets: uniqueOrdered(targets),
	}
}

func NewDataSetFile(filePath string, targetColumn int, hasHeader bool) (*DataSet, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	return NewDataSetReader(file, targetColumn, hasHeader)
}

func NewDataSetReader(reader io.Reader, targetColumn int, hasHeader bool) (*DataSet, error) {
	readerCsv := csv.NewReader(reader)
	readerCsv.ReuseRecord = true

	ds := &DataSet{}

	rowIdx := -1
	for row, err := range rows(readerCsv) {
		rowIdx++

		if err != nil {
			return nil, err
		}

		if rowIdx == 0 && hasHeader {
			continue
		}

		if targetColumn < -1 || targetColumn >= len(row) {
			return nil, fmt.Errorf("target column %d is out of range %d", targetColumn, len(row))
		}

		targetIdx := targetColumn; if targetColumn == -1 {
			targetIdx = len(row) - 1
		}

		for colIdx, col := range row {
			if colIdx == targetIdx {
				ds.targets = append(ds.targets, col)
				continue
			}

			float, err := strconv.ParseFloat(col, 64)
			if err != nil {
				return nil, fmt.Errorf("row %d, col %d: %w", rowIdx+1, colIdx, err)
			}

			ds.featuresRow = append(ds.featuresRow, float)
		}

		ds.rows++
	}

	ds.uniqueTargets = uniqueOrdered(ds.targets)

	return ds, nil
}

func rows(r *csv.Reader) iter.Seq2[[]string, error] {
	return func(yield func([]string, error) bool) {
		for {
			row, err := r.Read()
			if errors.Is(err, io.EOF) {
				return
			}

			if !yield(row, err) || err != nil {
				return
			}
		}
	}
}

func uniqueOrdered(items []string) []string {
	seen := make(map[string]struct{})
	var res []string
	for _, s := range items {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		res = append(res, s)
	}
	return res
}
