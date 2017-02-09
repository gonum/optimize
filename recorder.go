// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/gonum/floats"
)

var printerHeadings = [...]string{
	"Iter",
	"Runtime",
	"FuncEvals",
	"Func",
	"GradEvals",
	"|Gradient|∞",
	"HessEvals",
}

const (
	printerBaseTmpl = "%9v  %16v  %9v  %22v" // Base template for headings and values that are always printed.
	printerGradTmpl = "  %9v  %22v"          // Appended to base template when loc.Gradient != nil.
	printerHessTmpl = "  %9v"                // Appended to base template when loc.Hessian != nil.
)

// Printer writes column-format output to the specified writer as the optimization
// progresses. By default, it writes to os.Stdout.
type Printer struct {
	Writer          io.Writer
	HeadingInterval int
	ValueInterval   time.Duration

	lastHeading int
	lastValue   time.Time
}

func NewPrinter() *Printer {
	return &Printer{
		Writer:          os.Stdout,
		HeadingInterval: 30,
		ValueInterval:   500 * time.Millisecond,
	}
}

func (p *Printer) Init() error {
	p.lastHeading = p.HeadingInterval              // So the headings are printed the first time.
	p.lastValue = time.Now().Add(-p.ValueInterval) // So the values are printed the first time.
	return nil
}

func (p *Printer) Record(loc *Location, op Operation, stats *Stats) error {
	if op != MajorIteration && op != InitIteration && op != PostIteration {
		return nil
	}

	// Print values always on PostIteration or when ValueInterval has elapsed.
	printValues := time.Since(p.lastValue) > p.ValueInterval || op == PostIteration
	if !printValues {
		// Return early if not printing anything.
		return nil
	}

	// Print heading when HeadingInterval lines have been printed, but never on PostIteration.
	printHeading := p.lastHeading >= p.HeadingInterval && op != PostIteration
	if printHeading {
		p.lastHeading = 1
	} else {
		p.lastHeading++
	}

	if printHeading {
		headings := "\n" + fmt.Sprintf(printerBaseTmpl, printerHeadings[0], printerHeadings[1], printerHeadings[2], printerHeadings[3])
		if loc.Gradient != nil {
			headings += fmt.Sprintf(printerGradTmpl, printerHeadings[4], printerHeadings[5])
		}
		if loc.Hessian != nil {
			headings += fmt.Sprintf(printerHessTmpl, printerHeadings[6])
		}
		_, err := fmt.Fprintln(p.Writer, headings)
		if err != nil {
			return err
		}
	}

	values := fmt.Sprintf(printerBaseTmpl, stats.MajorIterations, stats.Runtime, stats.FuncEvaluations, loc.F)
	if loc.Gradient != nil {
		values += fmt.Sprintf(printerGradTmpl, stats.GradEvaluations, floats.Norm(loc.Gradient, math.Inf(1)))
	}
	if loc.Hessian != nil {
		values += fmt.Sprintf(printerHessTmpl, stats.HessEvaluations)
	}
	_, err := fmt.Fprintln(p.Writer, values)
	if err != nil {
		return err
	}

	p.lastValue = time.Now()
	return nil
}

// Full Recorder records all of the evaluations that occur during an optimization
// run. If Operation is an Evaulation, FullRecorder records the x location and
// the corresponding field of Location. If Operation is a MajorIteration, or
// PostIteration, FullRecorder records the full value of the Location struct.
// Otherwise, FullRecorder records the operation type.
type FullRecorder struct {
	Writer io.Writer
}

func (f FullRecorder) Init() error {
	return nil
}

func (f FullRecorder) Record(loc *Location, op Operation, stats *Stats) error {
	r := RecordLocation{Op: op.String()}
	switch {
	case op.isEvaluation():
		r.Loc.X = loc.X
		switch op {
		default:
			panic("optimize: unknown evaluation operation")
		case FuncEvaluation:
			r.Loc.F = loc.F
		case GradEvaluation:
			r.Loc.Gradient = loc.Gradient
		case HessEvaluation:
			r.Loc.Hessian = loc.Hessian
		}
	case op == MajorIteration || op == PostIteration:
		r.Loc = *loc
	}
	if op.isEvaluation() {

	} else if op == MajorIteration {

	}
	b, err := json.MarshalIndent(r, "", "\t")
	if err != nil {
		return nil
	}
	_, err = f.Writer.Write(b)
	return err
}

// Read reads the stream written to by FullRecorder, and returns the history of
// the optimization run.
func (f FullRecorder) Read(r io.Reader) ([]RecordLocation, error) {
	d := json.NewDecoder(r)
	var records []RecordLocation
	var err error
	for {
		var r RecordLocation
		err = d.Decode(&r)
		if err != nil {
			break
		}
		records = append(records, r)
	}
	if err == io.EOF {
		return records, nil
	}
	return records, err
}

type RecordLocation struct {
	Op  string
	Loc Location
}
